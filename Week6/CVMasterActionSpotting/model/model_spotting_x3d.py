# model_spotting_x3d.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm

# Local imports
from model.modules import BaseRGBModel, step

################################################################################
# 1) Custom Head: only pools over H/W, preserving T
################################################################################
class X3DHeadPerFrame(nn.Module):
    """
    A custom head for X3D that yields per-frame predictions:
      - input shape:  [B, in_channels, T, H, W]
      - we do an average pool over H,W, leaving T alone => [B, in_channels, T, 1, 1]
      - then a 1×1×1 conv to map from in_channels -> (num_classes+1) => [B, (C), T, 1, 1]
      - final squeeze to [B, (C), T] -> permute -> [B, T, C].
    """

    def __init__(self, in_channels=2048, num_classes=13):
        super().__init__()
        self.pool_spatial = nn.AdaptiveAvgPool3d((None, 1, 1))  # (T remains), H->1, W->1
        self.conv_project = nn.Conv3d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=(1, 1, 1),
            bias=True
        )

    def forward(self, x):
        # x => [B, in_channels, T, H, W]
        # 1) average pool only over H/W
        x = self.pool_spatial(x)  # => [B, in_channels, T, 1, 1]

        # 2) conv => [B, num_classes, T, 1, 1]
        x = self.conv_project(x)

        # 3) squeeze H,W => [B, num_classes, T]
        x = x.squeeze(-1).squeeze(-1)

        # 4) permute => [B, T, num_classes]
        x = x.permute(0, 2, 1).contiguous()

        return x


################################################################################
# 2) Main X3D Model Class
################################################################################
class ModelX3D(BaseRGBModel):
    """
    A single X3D model that directly takes a clip [B, T, C, H, W]
    and outputs [B, T, num_classes+1] for per-frame classification.
    We remove the standard global spatiotemporal pooling and use our custom head.
    """

    class Impl(nn.Module):
        def __init__(self, args=None):
            super().__init__()
            # ------------------------------------------------------------------
            # 1) Parse feature_arch: e.g. "x3d_s", "x3d_m", or "x3d_l"
            # ------------------------------------------------------------------
            arch_parts = args.feature_arch.split('-')
            x3d_variant = arch_parts[0]  # e.g. "x3d_s"

            # If last part is digit => freeze that many param groups
            if len(arch_parts) > 1 and arch_parts[-1].isdigit():
                self._trainable_layers = int(arch_parts[-1])
            else:
                self._trainable_layers = 0

            print("Trainable Layers: ", self._trainable_layers)
            print(f"Loading X3D variant: {x3d_variant}, pretrained=True")

            # ------------------------------------------------------------------
            # 2) Load PyTorchVideo's X3D and remove the final global head
            # ------------------------------------------------------------------
            model = torch.hub.load(
                'facebookresearch/pytorchvideo',
                x3d_variant,
                pretrained=True
            )
            # By default, model.blocks[-1] is an X3DHead that does global pooling
            # We'll replace it with Identity => the trunk now ends with [B, out_dim, T, H, W]
            model.blocks[-1] = nn.Identity()
            print(model)

            # figure out how many channels come out of the trunk
            # for x3d_s => 432, x3d_m => 1024, x3d_l => 2048 (approx)
            # you can set up a dictionary or do something dynamic. We'll do a simple mapping:
            out_channels_map = {
                'x3d_xs': 432,
                'x3d_s': 432,
                'x3d_m': 1024,
                'x3d_l': 2048
            }
            if x3d_variant not in out_channels_map:
                raise ValueError(f"Unknown X3D variant: {x3d_variant}")
            in_channels = out_channels_map[x3d_variant]

            # 3) Add our custom partial-pooling head
            num_out = args.num_classes + 1  # +1 for background
            model.per_frame_head = X3DHeadPerFrame(in_channels=in_channels, num_classes=num_out)

            # Store the trunk (blocks) + custom head
            self._features = model

            # 4) Freeze backbone if needed (except final head)
            self._freeze_backbone(self._features, self._trainable_layers)

            # ------------------------------------------------------------------
            # 5) Data augmentations & normalization
            # ------------------------------------------------------------------
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(5)], p=0.25),
                T.RandomHorizontalFlip(),
            ])
            self.standarization = T.Compose([
                T.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
            ])

        def forward(self, x):
            """
            x: [B, T, C, H, W] in [0..255].
            We want [B, T, (num_classes+1)] as output.
            """
            # 1) Scale from [0..255] to [0..1]
            x = x / 255.0
            B, T, C, H, W = x.shape

            # 2) Data augmentation (for each sample in batch)
            if self.training:
                for i in range(B):
                    x[i] = self.augmentation(x[i])  # shape: [T, C, H, W]

            # 3) Standardize each sub-tensor
            for i in range(B):
                x[i] = self.standarization(x[i])  # shape: [T, C, H, W]

            # 4) Reorder to X3D's expected input: [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4).contiguous()  # => [B, C, T, H, W]

            # 5) Forward pass:
            #    - the trunk (model.blocks) => [B, out_channels, T, H, W]
            #    - the custom partial pool head => [B, T, num_classes+1]
            feats_3d = self._features(x)         # "trunk" part
            print(feats_3d.shape)
            logits = self._features.per_frame_head(feats_3d)  # "head"
            print(logits.shape)

            return logits  # shape => [B, T, num_classes+1]

        def _freeze_backbone(self, backbone, trainable_layers):
            """
            Freeze everything except the last 'trainable_layers' param groups.
            Our custom head is always trainable.
            """
            # gather all trunk params (excluding our custom head)
            trunk_params = list(backbone.blocks.parameters())
            # by default, freeze them all:
            for p in trunk_params:
                p.requires_grad = False

            # unfreeze the last 'trainable_layers' param groups if requested
            if trainable_layers > 0:
                for p in trunk_params[-trainable_layers:]:
                    p.requires_grad = True

        def print_stats(self):
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'Model params: {trainable} trainable / {total} total')

    ############################################################################
    #  Main wrapper init + epoch + predict
    ############################################################################
    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = ModelX3D.Impl(args=args)
        self._model.print_stats()
        self._args = args
        self._model.to(self.device)

        self._num_classes = args.num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):
        """
        If optimizer is None => inference mode; else train mode.
        We'll do cross-entropy with the extra background class => [B, T, num_classes+1].
        But your label shape might also be [B, T]. Adjust accordingly.
        """
        if optimizer is None:
            self._model.eval()
            inference = True
        else:
            self._model.train()
            inference = False
            optimizer.zero_grad()

        # Weighted cross-entropy
        weights = torch.tensor([1.0] + [5.0] * self._num_classes, dtype=torch.float32).to(self.device)
        epoch_loss = 0.0

        with torch.no_grad() if inference else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frames = batch["frame"].to(self.device).float()   # [B, T, C, H, W]
                label = batch["label"].to(self.device).long()    # typically [B, T]

                B, T = label.shape

                with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                    # logits => [B, T, num_classes+1]
                    logits = self._model(frames)                 # => (B, T, C+1)
                    logits = logits.view(B * T, self._num_classes+1)   # => (B*T, C+1)
                    label = label.view(-1)                             # => (B*T)

                    loss = F.cross_entropy(logits, label, reduction='mean', weight=weights)

                if not inference:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)
                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict(self, seq):
        """
        seq: [T, C, H, W] or [B, T, C, H, W]
        Returns: [B, T, num_classes+1] (per-frame softmax probabilities).
        """
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if seq.ndim == 4:  # single clip => add batch
            seq = seq.unsqueeze(0)  # => [B=1, T, C, H, W]

        seq = seq.to(self.device).float()
        self._model.eval()

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                # => (B, T, num_classes+1)
                logits = self._model(seq)
            # softmax along last dim => [B, T, num_classes+1]
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy()

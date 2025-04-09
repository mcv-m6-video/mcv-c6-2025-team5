# model_spotting_x3d.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm

# Local imports
from model.modules import BaseRGBModel, FCLayers, step

class ModelX3D(BaseRGBModel):
    """
    A single X3D model that directly takes a clip [B, T, C, H, W]
    and outputs [B, T, num_classes+1] for per-frame classification,
    by removing the final global pooling over time.
    """

    class Impl(nn.Module):
        def __init__(self, args=None):
            super().__init__()

            # ------------------------------------------------------------------
            # 1) Parse feature_arch to figure out which X3D variant to load
            # ------------------------------------------------------------------
            arch_parts = args.feature_arch.split('-')
            x3d_variant = arch_parts[0]  # e.g. "x3d_s"

            if len(arch_parts) > 1 and arch_parts[-1].isdigit():
                self._trainable_layers = int(arch_parts[-1])
            else:
                self._trainable_layers = 0

            print("Trainable Layers: ", self._trainable_layers)
            print(f"Loading X3D variant: {x3d_variant}, pretrained=True")

            # ------------------------------------------------------------------
            # 2) Load X3D from PyTorchVideo Hub
            # ------------------------------------------------------------------
            model = torch.hub.load(
                'facebookresearch/pytorchvideo',
                x3d_variant,
                pretrained=True
            )

            # (A) Modify the X3D head to keep the time dimension
            head_block = model.blocks[-1]  # X3DHead
            out_dim = head_block.proj.in_features

            # Only pool over height & width => keeps time intact
            head_block.pool = nn.AdaptiveAvgPool3d((None, 1, 1))
            # Replace final linear layer with a 1x1x1 conv => maps [out_dim -> num_classes+1]
            head_block.proj = nn.Conv3d(
                in_channels=out_dim,
                out_channels=args.num_classes + 1,
                kernel_size=(1, 1, 1),
                bias=True
            )
            # If there's a built-in flatten or similar, remove/replace it
            if hasattr(head_block, 'output_pool'):
                head_block.output_pool = nn.Identity()

            self._features = model

            # Freeze backbone if needed
            self._freeze_backbone(self._features, self._trainable_layers)

            # ------------------------------------------------------------------
            # 3) Data augmentations & normalization
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
            We want to return [B, T, (num_classes+1)].
            """
            # 1) Scale from [0..255] -> [0..1]
            x = x / 255.0

            B, T, C, H, W = x.shape

            # 2) Data augmentation for each sample
            if self.training:
                for i in range(B):
                    x[i] = self.augmentation(x[i])  # shape [T, C, H, W]

            # 3) Standardize each sub-tensor
            for i in range(B):
                x[i] = self.standarization(x[i])  # [T, C, H, W]

            # 4) Reorder to X3D's expected input format [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]

            # 5) Forward pass in X3D => shape [B, (num_classes+1), T, 1, 1]
            logits_3d = self._features(x)

            # 6) Squeeze the spatial dims => [B, (num_classes+1), T]
            logits_3d = logits_3d.squeeze(dim=3).squeeze(dim=3)  # => [B, (num_classes+1), T]

            # 7) Permute to [B, T, (num_classes+1)] => one prediction per time index
            logits = logits_3d.permute(0, 2, 1).contiguous()

            return logits  # [B, T, num_classes+1]

        def _freeze_backbone(self, backbone, trainable_layers):
            """
            Freeze everything except the last 'trainable_layers' param groups.
            """
            all_params = list(backbone.parameters())
            for p in all_params:
                p.requires_grad = False
            if trainable_layers > 0:
                for p in all_params[-trainable_layers:]:
                    p.requires_grad = True

        def print_stats(self):
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'Model params: {trainable} trainable / {total} total')

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
        else:
            self._model.train()
            optimizer.zero_grad()

        # Weighted cross-entropy
        weights = torch.tensor([1.0] + [5.0] * self._num_classes, dtype=torch.float32).to(self.device)
        epoch_loss = 0.0

        with (torch.no_grad() if optimizer is None else nullcontext()):
            for batch_idx, batch in enumerate(tqdm(loader)):
                frames = batch["frame"].to(self.device).float()   # [B, T, C, H, W]
                label = batch["label"].to(self.device).long()    # [B, T] ideally

                # If your dataset has shape [B, T], you can flatten or handle per-frame:
                B, T = label.shape

                logits = None
                loss = None

                with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                    # logits => [B, T, num_classes+1]
                    logits = self._model(frames).view(B*T, self._num_classes+1)
                    label = label.view(-1)  # [B*T]
                    loss = F.cross_entropy(logits, label, reduction='mean', weight=weights)

                if optimizer is not None:
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
            seq = seq.unsqueeze(0)

        seq = seq.to(self.device).float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                # => (B, T, num_classes+1)
                logits = self._model(seq)
            # softmax along classes
            probs = torch.softmax(logits, dim=-1)  # => [B, T, num_classes+1]
        return probs.cpu().numpy()

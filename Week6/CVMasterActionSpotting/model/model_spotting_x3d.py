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
    and outputs [B, num_classes+1] for clip-level classification,
    using PyTorchVideo's X3D from torch.hub.
    """

    class Impl(nn.Module):
        def __init__(self, args=None):
            super().__init__()
            # ------------------------------------------------------------------
            # 1) Parse feature_arch to figure out which X3D variant to load
            #    and how many layers to unfreeze.
            #    E.g., if feature_arch="x3d_s_3", we interpret x3d_variant = "x3d_s"
            #    and trainable_layers = 3. Adjust to match your naming scheme.
            # ------------------------------------------------------------------
            arch_parts = args.feature_arch.split('-')
            x3d_variant = arch_parts[0]  # e.g. "x3d_s"
            
            # Check if last part is digit => number of trainable layers
            if len(arch_parts) > 1 and arch_parts[-1].isdigit():
                self._trainable_layers = int(arch_parts[-1])
            else:
                self._trainable_layers = 0

            print("Trainable Layers: ", self._trainable_layers)

            # ------------------------------------------------------------------
            # 2) Load X3D from PyTorchVideo Hub
            #    For instance: "x3d_s", "x3d_m", "x3d_l"
            # ------------------------------------------------------------------
            print(f"Loading X3D variant: {x3d_variant}, pretrained=True")
            model = torch.hub.load(
                'facebookresearch/pytorchvideo', 
                x3d_variant, 
                pretrained=True
            )

            print(model) 

            # The final classification layer is typically model.head.projection,
            # which we remove and replace with our own:
            head_block = model.blocks[-1]
            out_dim = head_block.proj.in_features
            head_block.proj = nn.Linear(out_dim, args.num_classes+1, bias=True)
            head_block.output_pool = nn.Identity()
            print(head_block)

            self._features = model
            self._freeze_backbone(self._features, self._trainable_layers)

            # ------------------------------------------------------------------
            # 4) Data augmentations & normalization
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
            Returns: [B, num_classes+1].
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

            # 5) Forward pass in X3D => global pool => [B, out_dim]
            logits = self._features(x)
            print(logits.shape)

            return logits

        def _freeze_backbone(self, backbone, trainable_layers):
            """
            Freeze all layers except the last 'trainable_layers' param groups.
            If trainable_layers=0, everything is frozen except the final FC we replaced.
            If trainable_layers is large, unfreeze more.
            """
            all_params = list(backbone.parameters())
            # Freeze everything
            for p in all_params:
                p.requires_grad = False
            # Unfreeze the last 'trainable_layers' param groups
            if trainable_layers > 0:
                for p in all_params[-trainable_layers:]:
                    p.requires_grad = True

        def print_stats(self):
            # Count trainable vs total
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f'Model params: {trainable} (trainable) out of {total} total.')

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
        If optimizer is None => inference mode. Otherwise, train mode.
        We'll do cross-entropy with an extra background class => [B, num_classes+1].
        Each sample is a single clip => label in [0..num_classes].
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
                frames = batch["frame"].to(self.device).float()  # [B, T, C, H, W]
                label = batch["label"].to(self.device).long()    # e.g. [B], or [B, T]=[B,1]

                with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                    logits = self._model(frames)  # [B, num_classes+1]
                    loss = F.cross_entropy(logits, label, reduction='mean', weight=weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)
                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict(self, seq):
        """
        seq: [T, C, H, W] or [B, T, C, H, W]
        Returns: [B, num_classes+1] softmax probabilities
        """
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if seq.ndim == 4:  # single clip
            seq = seq.unsqueeze(0)  # => [B=1, T, C, H, W]
        seq = seq.to(self.device).float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device == "cuda")):
                logits = self._model(seq)  # [B, num_classes+1]
            probs = torch.softmax(logits, dim=-1)  # [B, num_classes+1]
        return probs.cpu().numpy()

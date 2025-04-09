# model_spotting_x3d.py

import torch
from torch import nn
import timm
import torchvision.transforms as T
import torch.nn.functional as F
from contextlib import nullcontext
from tqdm import tqdm
import math

# Local imports
from model.modules import BaseRGBModel, FCLayers, step


class ModelX3D(BaseRGBModel):
    """
    A single X3D model that directly takes a clip [B, T, C, H, W]
    and outputs [B, num_classes+1] for clip-level classification.
    """

    class Impl(nn.Module):
        def __init__(self, args=None):
            super().__init__()

            # ---- 1) Parse feature_arch for X3D variant & create model  ----
            # For example, feature_arch might be "x3d_s", "x3d_m", "x3d_l", etc.
            self._feature_arch = args.feature_arch
            # We'll assume you have something like "x3d_s_3train" meaning "x3d_s" with 3 trainable layers, or
            # interpret however your config is set up. 
            # Example: "x3d_s" => small, "x3d_m" => medium, etc.

            # If you want to parse an underscore to get the trainable layers, do it here:
            arch_parts = self._feature_arch.split('-')
            x3d_variant = arch_parts[0]  # e.g. "x3d_s"
            if len(arch_parts) > 1 and arch_parts[-1].isdigit():
                self._trainable_layers = int(arch_parts[-1])  # e.g. 3
            else:
                self._trainable_layers = 0  # if not specified, 0 => freeze everything except final fc

            # Create the TIMM model
            # timm has "x3d_s", "x3d_m", etc. for X3D variants
            self._features = timm.create_model(
                x3d_variant, 
                pretrained=True
            )
            # By default, timm X3D does global average pooling over (space,time) and 
            # produces shape [B, out_dim]. We remove the classifier for our own head:
            out_dim = self._features.classifier.in_features
            self._features.classifier = nn.Identity()

            # ---- 2) Partial Finetuning: freeze everything except the last N layers  ----
            self._freeze_backbone(self._features, self._trainable_layers)

            # ---- 3) Final classification layer [B, out_dim] -> [B, num_classes+1]  ----
            self._fc = FCLayers(out_dim, args.num_classes + 1)

            # ---- 4) Data augmentation & normalization (same as before)  ----
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
            x: [B, T, C, H, W], 8-16 frames typically
            Returns: [B, T(?), num_classes+1] or [B, num_classes+1] 
                     Depending on your intended structure. 
            Here we do a single clip-level classification => [B, num_classes+1].
            """

            # 1) Scale from [0..255] to [0..1]
            x = x / 255.0

            # 2) Possibly apply augmentation (per-clip, but done frame by frame).
            B, T, C, H, W = x.shape
            if self.training:
                for i in range(B):
                    x[i] = self.augmentation(x[i])  # shape [T, C, H, W]

            # 3) Standardize each frame 
            #    (Note: we do it frame by frame. That is consistent with your existing code.)
            for i in range(B):
                x[i] = self.standarization(x[i])

            # 4) Reorder to X3D input format: [B, C, T, H, W]
            x = x.permute(0, 2, 1, 3, 4).contiguous()

            # 5) Forward pass through X3D (global pool in space & time by default => [B, out_dim])
            feats = self._features(x)  # shape [B, out_dim]

            # 6) Classification => [B, num_classes+1]
            logits = self._fc(feats)  

            return logits

        def _freeze_backbone(self, backbone, trainable_layers):
            """
            Freeze all layers except the last 'trainable_layers' parameter sets.
            If trainable_layers=0, everything is frozen except the newly added final FC.
            If trainable_layers is large, you'll unfreeze more.
            """
            all_params = list(backbone.parameters())
            # First, freeze everything
            for p in all_params:
                p.requires_grad = False
            # Unfreeze the last 'trainable_layers' param groups
            if trainable_layers > 0:
                for p in all_params[-trainable_layers:]:
                    p.requires_grad = True

        def print_stats(self):
            print('Model params:',
                  sum(p.numel() for p in self.parameters() if p.requires_grad),
                  '(trainable only, out of total:',
                  sum(p.numel() for p in self.parameters()), ')')

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
        If optimizer is not None, we are training. Otherwise, inference mode.
        We'll do cross-entropy with an extra background class => [B, num_classes+1].
        For each clip: label is a single integer in [0..num_classes].
        """

        if optimizer is None:
            self._model.eval()
        else:
            self._model.train()
            optimizer.zero_grad()

        # Weights for cross-entropy (background=1.0, each real class=5.0, for example)
        weights = torch.tensor([1.0] + [5.0] * (self._num_classes), dtype=torch.float32).to(self.device)

        epoch_loss = 0.0
        with (torch.no_grad() if optimizer is None else nullcontext()):
            for batch_idx, batch in enumerate(tqdm(loader)):
                frames = batch['frame'].to(self.device).float()  # [B, T, C, H, W]
                label = batch['label'].to(self.device).long()    # [B, T] or [B], depends on how the dataset is shaped.

                # If your dataset is shaped [B, T], but we are doing one label per clip,
                # you might do label = label[:,0] (just pick first, or check that T=1).
                # Adjust as needed based on your dataset. For a single clip label, you want [B].
                if len(label.shape) > 1: 
                    # If shape is [B, T], make sure we have T=1 or something appropriate
                    label = label[:, 0]  # example: just the first
                    # Or some other logic if your dataset is actually multi-frame labeled.

                with torch.cuda.amp.autocast(enabled=(self.device=='cuda')):
                    pred = self._model(frames)  # [B, num_classes+1]
                    loss = F.cross_entropy(pred, label, reduction='mean', weight=weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict(self, seq):
        """
        seq is either [T, C, H, W] or [B, T, C, H, W].
        Returns the softmax probabilities for each class: shape [B, num_classes+1].
        """
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4:  # (T, C, H, W)
            seq = seq.unsqueeze(0)  # => [B=1, T, C, H, W]
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device=='cuda')):
                logits = self._model(seq)  # [B, num_classes+1]
            probs = torch.softmax(logits, dim=-1)  # [B, num_classes+1]
            return probs.cpu().numpy()

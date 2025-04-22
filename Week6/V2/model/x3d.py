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

class X3DHead2(nn.Module):
    """
    A custom head for X3D that yields per-frame predictions:
      - input shape:  [B, in_channels, T, H, W]
      - we do an average pool over H,W, leaving T alone => [B, in_channels, T, 1, 1]
      - then a 1×1×1 conv to map from in_channels -> (num_classes+1) => [B, (C), T, 1, 1]
      - final squeeze to [B, (C), T] -> permute -> [B, T, C].
    """
    class ProjectedPool(nn.Module):
        def __init__(self, in_channels=192, out_channels=2048):
            super().__init__()
            self.pre_conv = nn.Conv3d(in_channels, 432, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            self.pre_norm = nn.BatchNorm3d(432, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            self.pre_act = nn.ReLU()
            self.pool = nn.AvgPool3d(kernel_size=(1, 5, 5), stride=1, padding=0)
            self.post_conv = nn.Conv3d(432, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
            self.post_act = nn.ReLU()

        def forward(self, x):
            x = self.pre_conv(x)
            x = self.pre_norm(x)
            x = self.pre_act(x)
            x = self.pool(x)
            x = self.post_conv(x)
            x = self.post_act(x)
            return x

    def __init__(self, in_channels=192, num_classes=13):
        super().__init__()
        self.pool = self.ProjectedPool(in_channels, 2048)
        self.dropout = nn.Dropout(p=0.5, inplace=False)
        self.proj = nn.Linear(in_features=2048, out_features=num_classes, bias=True)
        self.output_pool = nn.AdaptiveAvgPool3d(output_size=(None, 1, 1))

    def forward(self, x):
        x = self.pool(x)
        x = self.dropout(x)
        x = self.output_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = x.permute(0, 2, 1).contiguous()
        x = self.proj(x)
        return x

################################################################################
# 2) Main X3D Model Class
################################################################################
    

class X3DImpl(nn.Module):
    """
    A single X3D model that directly takes a clip [B, T, C, H, W]
    and outputs [B, T, num_classes+1] for per-frame classification.
    We remove the standard global spatiotemporal pooling and use our custom head.
    """
    def __init__(self, args=None):
        super().__init__()
        # ------------------------------------------------------------------
        # 1) Parse feature_arch: e.g. "x3d_s", "x3d_m", or "x3d_l"
        # ------------------------------------------------------------------
        x3d_variant, head = args.feature_arch.split('-')

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

        # figure out how many channels come out of the trunk
        # for x3d_s => 432, x3d_m => 1024, x3d_l => 2048 (approx)
        # you can set up a dictionary or do something dynamic. We'll do a simple mapping:
        out_channels_map = {
            'x3d_xs': 192,
            'x3d_s': 192,
            'x3d_m': 192,
            'x3d_l': 192
        }
        if x3d_variant not in out_channels_map:
            raise ValueError(f"Unknown X3D variant: {x3d_variant}")
        in_channels = out_channels_map[x3d_variant]

        # 3) Add our custom partial-pooling head
        num_out = args.num_classes + 1  # +1 for background
        if head == 'head1':
            self._head = X3DHeadPerFrame(in_channels=in_channels, num_classes=num_out)
        elif head == 'head2':
            self._head = X3DHead2(in_channels=in_channels, num_classes=num_out)

        # Store the trunk (blocks) + custom head
        self._features = model

        print(model)

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
            T.Normalize(mean=(0.45, 0.45, 0.45),
                        std=(0.225, 0.225, 0.225))
        ])

    def forward(self, x):
        """
        x: [B, T, C, H, W] in [0..255].
        We want [B, T, (num_classes+1)] as output.
        """
        x = self.normalize(x) #Normalize to 0-1
        batch_size, clip_len, channels, height, width = x.shape #B, T, C, H, W

        if self.training:
            x = self.augment(x) #augmentation per-batch

        x = self.standarize(x) #standarization imagenet stats

        # 4) Reorder to X3D's expected input: [B, C, T, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # => [B, C, T, H, W]

        # 5) Forward pass:
        #    - the trunk (model.blocks) => [B, out_channels, T, H, W]
        #    - the custom partial pool head => [B, T, num_classes+1]
        feats_3d = self._features(x)         # "trunk" part
        logits = self._head(feats_3d)  # "head"

        return logits  # shape => [B, T, num_classes+1]
    
    def normalize(self, x):
        return x / 255.
    
    def augment(self, x):
        for i in range(x.shape[0]):
            x[i] = self.augmentation(x[i])
        return x

    def standarize(self, x):
        for i in range(x.shape[0]):
            x[i] = self.standarization(x[i])
        return x

    def print_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Model params: {trainable} trainable / {total} total')

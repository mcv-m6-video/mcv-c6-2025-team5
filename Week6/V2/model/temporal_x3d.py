import torch
import torch.nn as nn
import torchvision.transforms as T

class PrintShape(nn.Module):
    def __init__(self, layer, tag=""):
        super().__init__()
        self.layer = layer          # the real layer you want to use
        self.tag   = tag            # optional label for the printout

    def forward(self, *x, **kw):
        out = self.layer(*x, **kw)  # run the real layer
        print(f"{self.tag} →", tuple(out.shape))
        return out

class TemporalX3D(nn.Module):
    """TemporalX3D
    ----------------
    A wrapper around a *pre‑trained* X3D variant that produces **per‑frame**
    class logits with output shape **B × T × C** (where *C = num_classes + 1*).

    Key design points
    -----------------
    1. **Sliding 3‑frame window (stride 1) with *same‑length* output.**  
       We pad the video temporally with black frames so the number of
       windows equals the original sequence length *T*.
    2. **Single forward call.** All windows are concatenated along the
       batch dimension (`B·T` clips) and passed through the X3D backbone
       in one go.
    3. **Classifier patch.** The head’s `proj` layer is replaced to emit
       `num_classes + 1` logits; if `clip_len < 4` the fixed `(4,5,5)` pool
       is swapped for a global adaptive pool so short clips are legal.
    """

    def __init__(self, args):
        super().__init__()
        self.num_classes = args.num_classes + 1  # +1 background class

        # ──────────────────────────────────────────────────────────────
        # 1)  Load & patch the pretrained X3D backbone
        # ──────────────────────────────────────────────────────────────
        variant = args.feature_arch  # e.g. "x3d_s", "x3d_m" …
        self.backbone = torch.hub.load(
            "facebookresearch/pytorchvideo", variant, pretrained=True
        )

        head = self.backbone.blocks[-1]  # ResNetBasicHead
        head.pool.pool = nn.AvgPool3d(kernel_size=(1, 5, 5), stride=1, padding=0)

        in_feats = head.proj.in_features
        head.proj = nn.Linear(in_feats, self.num_classes, bias=True)
        head.proj = PrintShape(nn.Linear(in_feats, self.num_classes, bias=True), tag='our_linear')
        head.output_pool = nn.AdaptiveAvgPool3d(output_size=(None, 1, 1))
        # head.output_pool = PrintShape(nn.AdaptiveAvgPool3d(output_size=(None, 1, 1)), tag="our_pool")
        # ──────────────────────────────────────────────────────────────
        # 2)  Augmentations & normalisation (ImageNet / PyTorchVideo)
        # ──────────────────────────────────────────────────────────────
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
            T.RandomApply([T.GaussianBlur(5)], p=0.25),
            T.RandomHorizontalFlip(),
        ])
        self.standardise = T.Normalize(mean=(0.45, 0.45, 0.45),
                                       std=(0.225, 0.225, 0.225))

    # ────────────────────────────────────────────────────────────────────
    # Helper transforms
    # ────────────────────────────────────────────────────────────────────
    @staticmethod
    def _to_float(x: torch.Tensor) -> torch.Tensor:
        """Scale from uint8 [0‥255] to float [0‥1]."""
        return x.float().div_(255.0)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        B, T, C, H, W = x.shape
        for b in range(B):
            x[b] = self.augmentation(x[b])
        return x

    def _standardise(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        for b in range(B):
            x[b] = self.standardise(x[b])
        return x

    # ────────────────────────────────────────────────────────────────────
    # Forward pass
    # ────────────────────────────────────────────────────────────────────
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Parameters
        ----------
        video : torch.Tensor
            Shape **B × T × C × H × W** with values in `[0‥255]`.

        Returns
        -------
        logits : torch.Tensor
            Shape **B × T × (num_classes + 1)** — one logit vector per
            *original* frame.
        """
        B, T, C, H, W = video.shape

        # ── 1) Pre‑processing ─────────────────────────────────────────
        x = self._to_float(video)
        x = self._augment(x)
        x = self._standardise(x)

        # Change to (B, C, T, H, W) for easier temporal ops
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        # ── 4) Forward through X3D backbone ───────────────────────────
        out = self.backbone(x)              # (B·T) × (num_classes+1)
        logits = out.view(B, T, -1)

        return logits

    # ────────────────────────────────────────────────────────────────────
    # Utilities
    # ────────────────────────────────────────────────────────────────────
    def print_stats(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model params: {trainable} trainable / {total} total")

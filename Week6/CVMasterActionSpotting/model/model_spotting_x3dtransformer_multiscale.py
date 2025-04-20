"""
File containing the main model.
"""

#Standard imports
import torch
from torch import nn
import timm
import torch.hub # Added for pytorchvideo
import torch.nn.functional as F # Added for pooling
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F


#Local imports
from model.modules import BaseRGBModel, FCLayers, step
from model.losses import focal_loss_multi_class


class TemporalFeaturePyramid(nn.Module):
    def __init__(self, in_channels, num_scales=3):
        super(TemporalFeaturePyramid, self).__init__()
        self.num_scales = num_scales
        # Convolutional layers to process each scale
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
            for _ in range(num_scales)
        ])

    def forward(self, features):
        # Input features: [batch, channels, time]
        scales = [features]
        # Generate coarser scales with temporal pooling
        for i in range(1, self.num_scales):
            pooled = F.avg_pool1d(scales[-1], kernel_size=2, stride=2)
            scales.append(pooled)

        # Process each scale with a 1D convolution
        processed_scales = []
        for i, scale in enumerate(scales):
            processed = self.scale_convs[i](scale)
            processed_scales.append(processed)

        # Combine: upsample coarser scales and add to the finest scale
        combined = processed_scales[0]
        for i in range(1, self.num_scales):
            # Upsample to match the finest scale's temporal length
            upsampled = F.interpolate(
                processed_scales[i],
                size=features.size(2),
                mode='linear',
                align_corners=False
            )
            combined = combined + upsampled

        return combined

class Model(BaseRGBModel):


    class Impl(nn.Module): 
        def __init__(self, args=None): 
            super().__init__() 
            self._feature_arch = args.feature_arch # Keep original variable name

            # --- MODIFICATION START: Replace RegNetY block with X3D ---
            # Parse feature_arch for X3D variant and freezing config (e.g., "x3d_s", "x3d_m-5")
            arch_parts = self._feature_arch.split('-')
            x3d_variant = arch_parts[0]
            trainable_layers = 0
            if len(arch_parts) > 1 and arch_parts[-1].isdigit():
                trainable_layers = int(arch_parts[-1])

            # Load pretrained X3D model
            # print(f"Loading X3D: {x3d_variant}, Trainable blocks: {trainable_layers if trainable_layers > 0 else 'All'}") # Optional print
            try:
                features = torch.hub.load('facebookresearch/pytorchvideo', model=x3d_variant, pretrained=True)
            except Exception as e:
                print(f"ERROR: Failed loading X3D model '{x3d_variant}'. Is pytorchvideo installed?")
                raise e

            # Remove the original X3D head
            features.blocks[-1] = nn.Identity()

            # Determine output dimension (_d) from X3D trunk
            # Using common values - verify if necessary for specific pytorchvideo version
            out_channels_map = {'x3d_xs': 192, 'x3d_s': 192, 'x3d_m': 192, 'x3d_l': 192}
            if x3d_variant in out_channels_map:
                feat_dim = out_channels_map[x3d_variant]
            else:
                # Fallback attempt - might not be robust
                try:
                    feat_dim = features.blocks[-2].proj.out_channels # Heuristic guess
                    # print(f"Warning: Guessed X3D output channels for {x3d_variant}: {feat_dim}") # Optional print
                except AttributeError:
                    raise ValueError(f"Unknown X3D variant '{x3d_variant}' and failed to infer channels. Add to out_channels_map.")

            self._d = feat_dim # Set feature dimension
            self._features = features # Assign X3D trunk

            # Freeze backbone layers if specified
            # self._freeze_backbone(self._features, trainable_layers)
            # --- MODIFICATION END ---

            # --- Temporal Transformer ---
            self.max_seq_len = args.clip_len
            self.positional_encoding = nn.Parameter(torch.randn(1, self.max_seq_len, self._d)) # _d is now from X3D

            num_heads = args.num_heads_transformer
            num_layers = args.num_layers_transformer
            print("Feature dimension (_d):", self._d)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self._d,
                nhead=num_heads,
                dim_feedforward=self._d * 2,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


            # --- MLP for classification ---
            self._fc = FCLayers(self._d, args.num_classes + 1)

            # --- Augmentations ---
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
                T.RandomApply([T.ColorJitter(saturation=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(brightness=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.ColorJitter(contrast=(0.7, 1.2))], p=0.25),
                T.RandomApply([T.GaussianBlur(5)], p=0.25),
                T.RandomHorizontalFlip(),
            ])

            self.standarization = T.Compose([
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

        def forward(self, x):
            x = self.normalize(x)
            B, T, C, H, W = x.shape

            if self.training:
                x = self.augment(x)

            x = self.standarize(x)

            # 1. Permute for X3D: [B, T, C, H, W] -> [B, C, T, H, W]
            x_permuted = x.permute(0, 2, 1, 3, 4).contiguous()
            # 2. Feature extraction with X3D trunk: Output [B, _d, T, H', W']
            x3d_out = self._features(x_permuted)
            # 3. Spatial Average Pooling per frame: -> [B, _d, T, 1, 1]
            pooled = F.adaptive_avg_pool3d(x3d_out, (None, 1, 1))
            # 4. Reshape for Transformer: -> [B, T, _d]
            # Squeeze H', W' dims -> [B, _d, T]; Permute T and _d -> [B, T, _d]
            im_feat = pooled.squeeze(-1).squeeze(-1).permute(0, 2, 1).contiguous()
            
            # Add positional encoding
            pos_encoding = self.positional_encoding[:, :T, :]
            im_feat = im_feat + pos_encoding  # (B, T, D)

            # Temporal modeling
            im_feat = self.transformer(im_feat)  # (B, T, D)

            # Classification
            im_feat = self._fc(im_feat)  # (B, T, num_classes+1)

            return im_feat

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

        # --- ADDED METHOD ---
        def _freeze_backbone(self, backbone, trainable_blocks):
            """Freezes backbone layers except the last `trainable_blocks`."""
            # Internal helper method, added for X3D freezing functionality
            if not hasattr(backbone, 'blocks'): return # Safety check
            trunk_params = list(backbone.blocks.parameters())
            for p in trunk_params: p.requires_grad = False # Freeze all first
            if trainable_blocks > 0:
                learnable_params = []
                num_blocks_total = len(list(backbone.blocks))
                if isinstance(backbone.blocks, (nn.Sequential, nn.ModuleList)):
                    start_idx = max(0, num_blocks_total - trainable_blocks)
                    for i in range(start_idx, num_blocks_total):
                        learnable_params.extend(list(backbone.blocks[i].parameters()))
                else: # Fallback if structure is different
                    if trainable_blocks <= len(trunk_params): learnable_params = trunk_params[-trainable_blocks:]
                    else: learnable_params = trunk_params # Train all if requested > available
                for p in learnable_params: p.requires_grad = True
        # --- END ADDED METHOD ---

        def print_stats(self): # Original method
            print('Model params:', 
                sum(p.numel() for p in self.parameters() if p.requires_grad), ' trainable / ', # Added trainable count display
                sum(p.numel() for p in self.parameters()), ' total') # Keep original total count display

    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        weights = torch.tensor([1.0] + [5.0] * (self._num_classes), dtype=torch.float32).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).long()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1) # B*T, num_classes
                    label = label.view(-1) # B*T
                    loss = focal_loss_multi_class(
                            pred, label, reduction='mean',gamma=1.2, alpha=weights)
                    # loss = F.cross_entropy(pred, label, reduction='mean', weight = weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss,
                        lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq):

        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)

            # apply sigmoid
            pred = torch.softmax(pred, dim=-1)
            
            return pred.cpu().numpy()


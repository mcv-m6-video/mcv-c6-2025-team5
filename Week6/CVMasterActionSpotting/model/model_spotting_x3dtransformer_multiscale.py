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
from model.modules import BaseRGBModel, FCLayers, step, CrossAttentionWithResidual
from model.losses import focal_loss_multi_class


class TemporalFeaturePyramid(nn.Module):
    def __init__(self, num_scales=3, d_model=192, downtr_nhead=12, num_layers=3, uptr_nhead = 12, max_time=50):
        super(TemporalFeaturePyramid, self).__init__()
        self.num_scales = num_scales
        self.d_model = d_model
        self.max_time = max_time
        
        # Positional encodings for each scale
        self.positional_encodings = []
        for _ in range(num_scales):
            self.positional_encodings.append(
                nn.Parameter(torch.randn(1, max_time, d_model))
            )
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=downtr_nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            batch_first=True
        )
        
        # Transformer encoder for each scale
        self.transformers = nn.ModuleList([
            nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            for _ in range(num_scales)
        ])
        # self.downsamplers = nn.ModuleList()
        # for i in range(1, num_scales):  # finest scale does not downsample           
        #     downsampler_block = nn.Sequential(
        #         nn.Conv1d(
        #             in_channels=d_model,
        #             out_channels=d_model,
        #             kernel_size=3,
        #             stride=1,
        #             padding=1
        #         ),
        #         nn.MaxPool1d(kernel_size=2, stride=2),
        #         nn.GELU()
        #     )
        #     self.downsamplers.append(downsampler_block)

        # self.downsamplers_from_raw = nn.ModuleList()
        # for i in range(1, num_scales):  # finest scale does not downsample           
        #     downsampler_block = nn.Sequential(
        #         nn.Conv1d(
        #             in_channels=d_model,
        #             out_channels=d_model,
        #             kernel_size=3,
        #             stride=1,
        #             padding=1
        #         ),
        #         nn.MaxPool1d(kernel_size=2, stride=2*i),
        #         nn.GELU()
        #     )
        #     self.downsamplers_from_raw.append(downsampler_block)
        
        # Transposed convolution for upsampling coarser scales
        # This is for upsampling that doubles every stage
        # self.upsamplers = nn.ModuleList()
        # for i in range(num_scales - 1):  # No upsampling for finest scale
        #     self.upsamplers.append(
        #         nn.ConvTranspose1d(
        #             in_channels=d_model,
        #             out_channels=d_model,
        #             kernel_size=4,  # Common choice for upsampling
        #             stride=2,
        #             padding=1,
        #             output_padding=0  # Adjust if needed for exact time dimension
        #         )
        #     )
        self.CAs = nn.ModuleList([
            CrossAttentionWithResidual(d_model, uptr_nhead)
            for _ in range(num_scales - 1)
        ])

    def forward(self, features):
        # Input features: [batch, channels, time]
        scales = [features]
        # Generate coarser scales with temporal pooling
        time_dim = self.max_time
        for i in range(1, self.num_scales):
            # pooled = F.avg_pool1d(scales[-1], kernel_size=2, stride=2)
            # pooled = self.downsamplers[i-1](scales[-1])
            
            pooled = F.avg_pool1d(scales[0], kernel_size=2, stride=1+i)

            # pooled = F.avg_pool1d(scales[0], kernel_size=2, stride=2*i)
            # pooled = self.downsamplers_from_raw[i-1](scales[0])
            scales.append(pooled)

        # Process each scale
        processed_scales = []
        for i, scale in enumerate(scales):
            # Reshape: [batch, channels, time] -> [batch, time, channels]
            batch, channels, time = scale.shape
            
            scale = scale.permute(0, 2, 1)  # [batch, time, d_model]
            
            # Add positional encoding
            pos_encoding = self.positional_encodings[i][:, :time, :].to(scale.device)
            transformer_input = scale + pos_encoding
            
            # Apply transformer
            transformer_out = self.transformers[i](transformer_input)  # [batch, time, d_model]
            
            # Reshape back to [batch, d_model, time]
            transformer_out = transformer_out.permute(0, 2, 1)
            
            processed_scales.append(transformer_out)
            
        improved_embs = [processed_scales[0]]    
        for i in range(self.num_scales-1, 0, -1):
            # Compute target time dimension (original time)
            target_time = processed_scales[i-1].shape[2]
            scale = processed_scales[i]
            
            # Apply transposed convolution
            # upsampled = self.upsamplers[i-1](scale)  # [batch, d_model, ~time]
            
            # Trim or pad to match exact time dimension
            # current_time = upsampled.shape[-1]
            # if current_time > target_time:
            #     upsampled = upsampled[:, :, :target_time]
            # elif current_time < target_time:
            #     upsampled = F.pad(upsampled, (0, target_time - current_time))
                
            # improved = self.CAs[i-1](target = processed_scales[i-1].permute(0,2,1).contiguous(),
            #                    source = upsampled.permute(0,2,1).contiguous())
            # improved = self.CAs[i-1](target = processed_scales[i-1].permute(0,2,1).contiguous(),
            #                    source = scale.permute(0,2,1).contiguous())
            
            # improved = self.CA(target = upsampled.permute(0,2,1).contiguous(),
            #                    source = processed_scales[i-1].permute(0,2,1).contiguous())
            # processed_scales[i-1] = improved.permute(0,2,1).contiguous()

            improved = self.CAs[i-1](target = processed_scales[0].permute(0,2,1).contiguous(),
                               source = scale.permute(0,2,1).contiguous()) #convert to [B, T, D] for attention
            improved_embs.append(improved.permute(0,2,1).contiguous()) #convert back to [B, D, T]
            
        # return processed_scales[0]
        return torch.stack(improved_embs, dim=0).mean(dim=0)  # [batch, d_model, time]

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

            # --- Temporal Transformer ---
            self.max_seq_len = args.clip_len

            downtr_num_heads = args.num_heads_transformer
            downtr_num_layers = args.num_layers_transformer
            uptr_num_heads =  args.num_heads_crossattention
            self.pyramid = TemporalFeaturePyramid(num_scales=args.num_scales, d_model=self._d, uptr_nhead=uptr_num_heads, downtr_nhead=downtr_num_heads,
                                                  num_layers=downtr_num_layers, max_time=self.max_seq_len)

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
            # 4. Reshape
            im_feat = pooled.squeeze(-1).squeeze(-1).contiguous()

            # 5. Temporal modeling
            im_feat = self.pyramid(im_feat)  # [(B, D, T), (B, D, T/2), (B, D, T/4), ...]
            im_feat = im_feat.permute(0, 2, 1).contiguous()  # (B, T, D)
            # 6. Classification
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


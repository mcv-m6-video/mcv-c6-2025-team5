"""
File containing the main model.
"""

#Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F
import math 

#Local imports
from model.modules import BaseRGBModel, FCLayers, step

class Model(BaseRGBModel):


    class Impl(nn.Module):
        def __init__(self, args=None):
            super().__init__()
            self._feature_arch = args.feature_arch
            # --- Configuration ---
            self.stack_size = getattr(args, 'stack_size', 3) # Number of frames to stack (k)

            # --- Feature Extractor (RegNetY) ---
            # Expects 3 input channels. Since we stack k=3 grayscale frames,
            # the input dimensions match standard pre-trained models.
            if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True, in_chans=self.stack_size) # Ensure in_chans matches stack_size
                feat_dim = features.head.fc.in_features
                features.head.fc = nn.Identity()
                self._d = feat_dim
                print(f"Feature dim: {self._d}")
            else:
                raise NotImplementedError(args.feature_arch) # Corrected attribute access

            self._features = features

            # --- Temporal Transformer ---
            # Max sequence length now refers to the number of groups/stacks
            self.max_original_seq_len = args.clip_len
            self.max_seq_len_groups = self.max_original_seq_len // self.stack_size
            print(f"Using stack size: {self.stack_size}. Max original frames: {self.max_original_seq_len}. Max groups for Transformer: {self.max_seq_len_groups}")

            # Positional encoding for the sequence of groups
            self.positional_encoding = nn.Parameter(torch.randn(1, self.max_seq_len_groups, self._d))

            num_heads = args.num_heads_transformer
            num_layers = args.num_layers_transformer

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self._d,
                nhead=num_heads,
                dim_feedforward=self._d * 2,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


            # --- MLP for classification ---
            # Output classes per group
            self._fc = FCLayers(self._d, args.num_classes + 1)

            # --- Augmentations ---
            # Apply augmentations suitable for RGB *before* grayscale conversion
            # Remove ColorJitter
            self.augmentation = T.Compose([
                T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.25),
                T.RandomHorizontalFlip(),
            ])

            # --- Preprocessing ---
            self.grayscale = T.Grayscale(num_output_channels=1)
            # Standard normalization - applied after stacking grayscale frames
            grayscale_mean = 0.44531356896770125
            grayscale_std = 0.2692461874154524
            self.standardization = T.Compose([
                T.Normalize(
                    mean=(grayscale_mean ,grayscale_mean, grayscale_mean), 
                    std=(grayscale_std, grayscale_std, grayscale_std)
                )
            ])


        def forward(self, x):
            # x shape: (B, T, C, H, W) - Assuming C=3 initially
            B, T, C_in, H, W = x.shape
            x = self.normalize(x) # Normalize to [0, 1] first

            # Apply augmentations (spatial) before grayscale conversion
            if self.training:
                x = self.augment(x) # Operates on (B, T, C, H, W)

            # --- Grayscale Conversion ---
            # Reshape to process frames individually
            x_reshaped = x.view(B * T, C_in, H, W)
            # Apply grayscale
            x_gray = self.grayscale(x_reshaped) # Shape: (B*T, 1, H, W)
            x_gray = x_gray.view(B, T, 1, H, W)

            # --- Frame Stacking ---
            # Truncate T to be divisible by stack_size
            num_groups = T // self.stack_size
            if num_groups == 0:
                 raise ValueError(f"Input sequence length {T} is less than stack size {self.stack_size}")
            T_trunc = num_groups * self.stack_size
            x_gray = x_gray[:, :T_trunc, :, :, :] # Shape: (B, T_trunc, 1, H, W)

            # Reshape and stack along channel dimension
            # (B, T_trunc, 1, H, W) -> (B, num_groups, stack_size, H, W)
            x_stacked = x_gray.view(B, num_groups, self.stack_size, H, W)

            # Reshape for Conv2D: (B, num_groups, stack_size, H, W) -> (B * num_groups, stack_size, H, W)
            x_conv_input = x_stacked.reshape(B * num_groups, self.stack_size, H, W)

            # Apply standardization (treating stacked gray channels like RGB)
            x_conv_input = self.standardize(x_conv_input) # Operates on (B*T', k, H, W)

            # --- Feature extraction: Shared CNN per stack ---
            # (B * num_groups, stack_size, H, W) -> (B * num_groups, D)
            im_feat = self._features(x_conv_input)

            # Reshape for Transformer: (B * num_groups, D) -> (B, num_groups, D)
            im_feat = im_feat.view(B, num_groups, self._d)

            # --- Temporal Modeling ---
            # Add positional encoding for the groups
            # Ensure positional encoding length matches or exceeds num_groups
            if num_groups > self.max_seq_len_groups:
                 print(f"Warning: Input sequence has {num_groups} groups, but max positional encoding is {self.max_seq_len_groups}. Truncating sequence for positional encoding.")
                 im_feat = im_feat[:, :self.max_seq_len_groups, :]
                 current_num_groups = self.max_seq_len_groups
            else:
                 current_num_groups = num_groups

            pos_encoding = self.positional_encoding[:, :current_num_groups, :]
            im_feat = im_feat + pos_encoding  # (B, current_num_groups, D)

            # Apply Transformer
            im_feat = self.transformer(im_feat)  # (B, current_num_groups, D)

            # --- Classification ---
            # Output shape: (B, current_num_groups, num_classes+1)
            group_preds = self._fc(im_feat)

            # Note: The output temporal dimension now corresponds to groups, not original frames.
            # Label processing in the `epoch` method needs to account for this if labels are per-frame.
            frame_preds = torch.repeat_interleave(group_preds, repeats=self.stack_size, dim=1)
            return frame_preds

        def normalize(self, x):
            # Assuming input x is B, T, C, H, W and 0-255
            return x / 255.

        def augment(self, x):
            # Input x: (B, T, C, H, W)
            # Apply augmentation to each frame
            B, T, C, H, W = x.shape
            x_flat = x.view(B * T, C, H, W)
            x_aug = self.augmentation(x_flat) # Apply T.Compose to (N, C, H, W)
            return x_aug.view(B, T, C, H, W)

        def standardize(self, x):
            # Input x: (N, C, H, W) - where N = B * num_groups, C = stack_size
            # Apply standardization per item in the batch N
            # Ensure input is float for normalization
            x = x.float()
            return self.standardization(x) # T.Compose handles batch application implicitly


        def print_stats(self):
            print('Model params:',
                sum(p.numel() for p in self.parameters()))

    # --- Wrapper class methods (epoch, predict, __init__) ---
    # Need adjustments mainly in label handling within 'epoch' if labels are per-frame.
    # The 'predict' method should work as is, returning predictions per group.

    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and hasattr(args, 'device') and (args.device == "cuda"):
            self.device = "cuda"

        self._model = Model.Impl(args=args)
        self._model.print_stats()
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes
        self.stack_size = getattr(args, 'stack_size', 3) # Store stack_size

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        # Weights remain the same logic
        weights = torch.tensor([1.0] + [5.0] * (self._num_classes), dtype=torch.float32).to(self.device)

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device) # Should be float if coming directly
                label = batch['label'] # Original labels, likely (B, T)
                label = label.to(self.device).long()

                # --- Label Adaptation (CRITICAL) ---
                # The model outputs predictions per group (B, T // stack_size, num_classes+1)
                # Original labels are likely per frame (B, T)
                # We need to adapt labels to match the prediction's temporal dimension.
                # Strategy: Take the label of the middle frame in each stack? Or max pool labels?
                # Example: Take label of middle frame (adjust index carefully)
                B, T_orig = label.shape
                num_groups = T_orig // self.stack_size
                T_trunc = num_groups * self.stack_size
                label = label[:, :T_trunc] # Truncate labels like input frames

                # Ensure adapted_label shape matches flattened prediction shape later
                adapted_label_flat = label.reshape(-1) # Shape: (B * num_groups)
                # --- End Label Adaptation ---


                with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                    pred = self._model(frame) # Output: (B, T_trunc, num_classes+1)
                    # Flatten prediction for loss calculation
                    pred_flat = pred.view(-1, self._num_classes + 1) # Shape: (B * T_trunc, num_classes+1)

                    # Ensure shapes match (prediction vs label)
                    # This should now match if T_trunc is handled consistently
                    if pred_flat.shape[0] != adapted_label_flat.shape[0]:
                        min_len = min(pred_flat.shape[0], adapted_label_flat.shape[0])
                        print(f"Warning: Mismatch between flattened prediction ({pred_flat.shape[0]}) and label ({adapted_label_flat.shape[0]}) lengths. Using min_len: {min_len}")
                        pred_flat = pred_flat[:min_len, :]
                        adapted_label_flat = adapted_label_flat[:min_len]

                    loss = F.cross_entropy(
                            pred_flat, adapted_label_flat, reduction='mean', weight=weights)

                if optimizer is not None:
                    step(optimizer, scaler, loss,
                        lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)     # Avg loss

    def predict(self, seq):

        if not isinstance(seq, torch.Tensor):
            # Assuming seq is numpy HWC format or similar, convert to BCHW Tensor
             # This part might need adjustment based on actual input format
             # Let's assume input is already (T, C, H, W) or (B, T, C, H, W) tensor for simplicity here
            seq = torch.FloatTensor(seq) # Simplification
        if len(seq.shape) == 4: # (T, C, H, W)
            seq = seq.unsqueeze(0) # Add Batch dimension -> (1, T, C, H, W)
        if seq.device != self.device:
            seq = seq.to(self.device)
        # Input expected by forward is B, T, C, H, W
        # Ensure input is float, normalization happens inside forward
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            # Mixed precision inference recommended if scaler was used in training
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred = self._model(seq) # Output: (B, num_groups, num_classes+1)

            # Apply softmax
            pred = torch.softmax(pred, dim=-1)

            # Return probabilities per group
            return pred.cpu().numpy() # Shape (B, num_groups, num_classes+1)
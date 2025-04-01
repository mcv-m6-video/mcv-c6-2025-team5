import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F

# Local imports
from model.modules import BaseRGBModel, FCLayers, step
from model.losses import BinaryFocalLoss

############################
# 1) Simple Attn Pooling
############################

class TemporalAttention(nn.Module):
    """
    A simple attention pooling that uses a learned query to attend over T frame features.
    Produces a single B x D vector per clip.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # The "query" vector that will attend to the frame embeddings
        self.query = nn.Parameter(torch.randn(hidden_dim))  # shape (D,)

        # We'll create a small MLP to generate attention logits:
        # attends to the dot(query, frame_feature) or we can do additive
        self.mlp = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape

        # We'll do a simple 'query' mechanism:
        # 1) expand query => shape (B, T, D) so we can do elementwise
        q = self.query.view(1, 1, D).expand(B, T, D)  # (B, T, D)

        # 2) e.g. dot or concat. For simplicity, let's do an elementwise multiply and then linear
        #    or just do: attn_logits = (x * q).sum(-1)
        #    Then pass it through an MLP or so. Let's keep it simple:

        attn_logits = self.mlp(x * q).squeeze(-1)  # => (B, T)

        # 3) Softmax over T
        attn_weights = torch.softmax(attn_logits, dim=-1)  # (B, T)

        # 4) Weighted sum
        attn_weights = attn_weights.unsqueeze(-1)          # (B, T, 1)
        out = (x * attn_weights).sum(dim=1)                # (B, D)

        return out


############################
# 2) Self-Attention Transformer
############################

class TemporalTransformer(nn.Module):
    """
    A small Transformer-based aggregator. We insert a CLS token, run self-attn,
    then take the CLS output as the clip-level vector.
    """
    def __init__(self, hidden_dim, num_layers=2, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim

        # A single learnable CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # A TransformerEncoder from PyTorch
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            dropout=0.1,
            batch_first=True  # input shape (B, S, D)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layernorm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: shape (B, T, D)
        B, T, D = x.shape

        # Expand CLS token across the batch
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)

        # Concat: (B, T+1, D)
        x = torch.cat([cls, x], dim=1)

        # Run the transformer
        x = self.transformer(x)  # (B, T+1, D)

        # Take the CLS output (index=0)
        cls_out = x[:, 0]  # (B, D)
        cls_out = self.layernorm(cls_out)

        return cls_out


class NewImpl(nn.Module):
    def __init__(self, args = None):
        super().__init__()
        self._feature_arch = args.feature_arch

        # 1) Parse aggregator from the string
        #    We expect something like "rny002_temporal" or "rny002_temporalSelfAttn"
        #    If there's no underscore, default to maxpool
        if '_' in self._feature_arch:
            base_arch, aggregator = self._feature_arch.split('_', 1)
        else:
            base_arch = self._feature_arch
            aggregator = 'maxpool'
        print(base_arch, aggregator)

        # 2) Build the RegNet backbone
        if base_arch in ['rny002','rny004','rny008']:
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[base_arch], pretrained=True)
            feat_dim = features.head.fc.in_features

            # Remove final classification layer
            features.head.fc = nn.Identity()
            self._d = feat_dim

        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        # 3) Build aggregator
        self._aggregator_type = aggregator.lower()
        if self._aggregator_type == 'maxpool':
            # We'll do the old approach (no extra params)
            self._aggregator = None
        elif self._aggregator_type == 'temporalattention':
            # We'll do the simple attention pooling
            self._aggregator = TemporalAttention(hidden_dim=self._d)
        elif self._aggregator_type == 'temporaltransformer':
            # We'll do a small transformer
            self._aggregator = TemporalTransformer(hidden_dim=self._d, num_layers=2, num_heads=4)
        else:
            print(f"Warning: aggregator {aggregator} not recognized. Defaulting to maxpool.")
            self._aggregator_type = 'maxpool'
            self._aggregator = None

        # MLP for classification
        self._fc = FCLayers(self._d, args.num_classes)

        # Augmentations
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue=0.2)], p=0.25),
            T.RandomApply([T.ColorJitter(saturation=(0.7,1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(brightness=(0.7,1.2))], p=0.25),
            T.RandomApply([T.ColorJitter(contrast=(0.7,1.2))], p=0.25),
            T.RandomApply([T.GaussianBlur(5)], p=0.25),
            T.RandomHorizontalFlip(),
        ])

        # Standardization
        self.standarization = T.Normalize(
            mean=(0.485,0.456,0.406), 
            std =(0.229,0.224,0.225))

    def forward(self, x):
        """
        x: shape (B, T, C, H, W)
        """
        x = self.normalize(x)  # Normalize to [0..1]
        B, T, C, H, W = x.shape

        if self.training:
            x = self.augment(x)

        # Imagenet stats
        # We'll do it frame-by-frame inside a loop or vectorized
        x = self.standarize_batch(x)

        # Flatten (B*T, C, H, W), run backbone
        feats = self._features(x.view(-1, C, H, W))  # (B*T, D)
        feats = feats.view(B, T, self._d)            # (B, T, D)

        # --- Aggregation ---
        if self._aggregator_type == 'maxpool' or self._aggregator is None:
            # Old approach
            clip_feat = torch.max(feats, dim=1)[0]   # (B, D)
        else:
            # Use the aggregator module
            clip_feat = self._aggregator(feats)      # (B, D)

        # FC => (B, num_classes)
        logits = self._fc(clip_feat)
        return logits

    def normalize(self, x):
        return x / 255.

    def augment(self, x):
        # x is shape (B, T, C, H, W)
        # We can apply the augmentation per sample
        for i in range(x.shape[0]):
            # x[i] => shape (T, C, H, W)
            x[i] = self.augmentation(x[i])
        return x

    def standarize_batch(self, x):
        # shape (B, T, C, H, W)
        B, T, C, H, W = x.shape
        # We can do: apply transform individually or flatten B*T dimension
        x = x.view(B*T, C, H, W)
        x = self.standarization(x)
        x = x.view(B, T, C, H, W)
        return x

    def print_stats(self):
        print('Model params:',
              sum(p.numel() for p in self.parameters()))


class NewModel(BaseRGBModel):
    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = NewImpl(args=args)
        self._model.print_stats()
        self._args = args
        self.loss = BinaryFocalLoss() if args.loss == "focal" else F.binary_cross_entropy_with_logits

        self._model.to(self.device)
        self._num_classes = args.num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):
        if optimizer is None:
            # Validation / inference
            self._model.eval()
        else:
            self._model.train()

        epoch_loss = 0.0

        # no_grad context only for inference (optimizer=None)
        grad_ctx = torch.no_grad() if (optimizer is None) else nullcontext()
        with grad_ctx:
            for batch_idx, batch in enumerate(tqdm(loader)):
                frames = batch['frame'].to(self.device).float()
                labels = batch['label'].to(self.device).float()

                with torch.cuda.amp.autocast(enabled=(self.device=="cuda")):
                    logits = self._model(frames)
                    loss = self.loss(logits, labels)

                if optimizer is not None:
                    step(optimizer, scaler, loss, lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)

    def predict(self, seq):
        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        # If shape is (T, C, H, W), make it (1, T, C, H, W)
        if len(seq.shape) == 4:
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self.device=="cuda")):
            logits = self._model(seq)
            preds = torch.sigmoid(logits)
        return preds.cpu().numpy()

    def get_optimizer(self, opt_args, finetune_lr_factor=1.0):
        """
        If you want to have a smaller LR for the backbone vs. classification head,
        e.g. pass finetune_lr_factor=0.1 or something in the call site.
        """
        # 1) Separate backbone vs head
        backbone_params = list(self._model._features.parameters())
        head_params = []
        # Also aggregator params (if it exists)
        if self._model._aggregator:
            head_params += list(self._model._aggregator.parameters())
        head_params += list(self._model._fc.parameters())

        # 2) Adjust LR
        #   If base LR is, say, 1e-4, backbone is 1e-4 * finetune_lr_factor, head is 1e-4
        backbone_lr = opt_args['lr'] * finetune_lr_factor
        head_lr     = opt_args['lr']

        param_groups = [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params,     'lr': head_lr},
        ]

        # remove 'lr' from opt_args because we just used it
        del opt_args['lr']

        optimizer = torch.optim.AdamW(param_groups, **opt_args)
        scaler = torch.cuda.amp.GradScaler() if self.device == 'cuda' else None
        return optimizer, scaler

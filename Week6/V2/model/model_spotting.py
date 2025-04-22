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
from model.losses import FocalLoss, SoftDiceLoss

from functools import partial

from util.io import save_images

from model.baseline import BaselineImpl
from model.x3d import X3DImpl
from model.temporal_x3d import TemporalX3D


#Local imports
from model.modules import BaseRGBModel, FCLayers, step

class Model(BaseRGBModel):
    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        if args.model_impl == 'baseline':
            self._model = BaselineImpl(args=args)
        elif args.model_impl == 'x3d':
            self._model = X3DImpl(args=args)
        elif args.model_impl == 'temporal_x3d':
            self._model = TemporalX3D(args=args)
        else:
            raise Exception("Wrong model")
        
        self._model.print_stats()
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes

        class_weights =  torch.tensor([args.background_weight] + [1.0]*(self._num_classes), dtype=torch.float32).to(self.device)
        class_weights = class_weights / class_weights.mean()
        
        if args.loss == 'focal':
            self._loss_fn = FocalLoss(
                self._num_classes+1,
                gamma=1.0,
                weight=class_weights,
                device=self.device)
        elif args.loss == 'ce':
            self._loss_fn = torch.nn.CrossEntropyLoss(
                weight = class_weights,
                reduction='mean')
        elif args.loss == 'dice':
            self._loss_fn = SoftDiceLoss()
        else:
            raise Exception("Invalid Loss")

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['target']
                label = label.to(self.device).float()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1) # B*T, num_classes
                    label = label.view(-1, self._num_classes + 1) # B*T
                    loss = self._loss_fn(pred, label)
                    # print("Pred: ", pred.shape)
                    # print("Label: ", label.shape, label)
                    # print("Loss: ", loss)

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

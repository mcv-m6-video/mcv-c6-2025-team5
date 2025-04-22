import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F
from model.losses import FocalLoss, SoftDiceLoss

from model.modules import BaseRGBModel, FCLayers, step

class BaselineImpl(nn.Module):
    def __init__(self, args = None):
        super().__init__()
        self._feature_arch = args.feature_arch

        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features

            # Remove final classification layer
            features.head.fc = nn.Identity()
            self._d = feat_dim

        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        # MLP for classification
        self._fc = FCLayers(self._d, args.num_classes+1) # +1 for background class (we now perform per-frame classification with softmax, therefore we have the extra background class)

        #Augmentations and crop
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
            T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
            T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
            T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
            T.RandomApply([T.GaussianBlur(5)], p = 0.25),
            T.RandomHorizontalFlip(),
        ])

        #Standarization
        self.standarization = T.Compose([
            T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
        ])

    def forward(self, x):
        x = self.normalize(x) #Normalize to 0-1
        batch_size, clip_len, channels, height, width = x.shape #B, T, C, H, W

        if self.training:
            x = self.augment(x) #augmentation per-batch

        x = self.standarize(x) #standarization imagenet stats
                    
        im_feat = self._features(
            x.view(-1, channels, height, width)
        ).reshape(batch_size, clip_len, self._d) #B, T, D

        #MLP
        im_feat = self._fc(im_feat) #B, T, num_classes+1

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

    def print_stats(self):
        print('Model params:',
            sum(p.numel() for p in self.parameters()))
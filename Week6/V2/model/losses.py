import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=2.0, weight=None, device='cpu'):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.device = device

        if weight is None:
            self.weights = torch.ones(num_classes, device=self.device)
        else:
            self.weights = weight.to(self.device)
        
        self.ce_loss = torch.nn.CrossEntropyLoss(
                weight = self.weights,
                reduction='none')

    def forward(self, pred, target):
        """
        pred: (N, C) logits
        target: (N, C) class probabilities
        """
        ce_loss = self.ce_loss(pred, target)
        pt = torch.exp(-ce_loss)
        focal_factor = (1 - pt) ** self.gamma
        loss = focal_factor * ce_loss
        return loss.mean()
    
class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, exclude_background=True):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.exclude_background = exclude_background

    def forward(self, logits, targets):
        """
        preds: Tensor of shape (N, C), predicted class probabilities (e.g. after softmax)
        targets: Tensor of shape (N, C), ground truth class probabilities (e.g. soft labels)
        """
        if self.exclude_background:
            preds = F.softmax(logits, dim=1)[:, 1:]
            targets = targets[:, 1:]
        else:
            preds = F.softmax(logits, dim=1)
        # Flatten over the batch (N) and classes (C) to compute per-sample per-class overlap
        intersection = torch.sum(preds * targets, dim=0)
        preds_sum = torch.sum(preds * preds, dim=0)
        targets_sum = torch.sum(targets * targets, dim=0)

        dice_score = (2 * intersection + self.smooth) / (preds_sum + targets_sum + self.smooth)
        dice_loss = 1 - dice_score

        # Return mean over classes
        return dice_loss.mean()

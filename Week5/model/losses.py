import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Binary focal loss for multi-label classification.
        
        Args:
            alpha (float): Weighting factor for the positive class (default: 0.25).
            gamma (float): Focusing parameter (default: 2.0).
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(BinaryFocalLoss, self).__init__()
        print("Using Focal Loss with alpha={} and gamma={}".format(alpha, gamma))
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Compute focal loss.
        
        Args:
            logits (torch.Tensor): Raw logits from the model (before sigmoid).
            targets (torch.Tensor): Binary ground truth labels (0 or 1).
        
        Returns:
            torch.Tensor: Focal loss.
        """
        # Compute the sigmoid of the logits to get probabilities
        p = torch.sigmoid(logits)
        
        # Compute binary cross-entropy terms
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Compute the probability of the true class (p_t)
        p_t = p * targets + (1 - p) * (1 - targets)
        
        # Compute the focal loss term: -(1 - p_t)^gamma * log(p_t)
        loss = ce_loss * (1 - p_t) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
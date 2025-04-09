import torch
import torch.nn.functional as F
from torch import nn

def focal_loss_multi_class(logits: torch.Tensor,
                           targets: torch.Tensor,
                           gamma: float = 2.0,
                           alpha: torch.Tensor = None, # Per-class weighting factor
                           reduction: str = 'mean') -> torch.Tensor:
    """
    Compute the Focal Loss for multi-class classification.
    Args:
        logits: Raw model outputs (logits) of shape (N, C).
        targets: Ground truth labels of shape (N,) where each value is 0 <= targets[i] < C.
        gamma: Focusing parameter (>= 0). Default: 2.0.
        alpha: Weighting factor for class imbalance, tensor of shape (C,). If None, no weighting.
        reduction: Specifies the reduction to apply: 'none' | 'mean' | 'sum'. Default: 'mean'.
    Returns:
        The calculated focal loss.
    """
    num_classes = logits.shape[1]
    if logits.device != targets.device:
        targets = targets.to(logits.device)

    # Calculate log probabilities_ true class for each example
    log_softmax_probs = F.log_softmax(logits, dim=1)

    # Gather the log probabilities corresponding to the true classes
    log_prob_targets = log_softmax_probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # Calculate the probability of the true class P(y) = P_t
    prob_targets = torch.exp(log_prob_targets)

    # Calculate the modulating factor (1 - P(y))^gamma
    modulating_factor = (1.0 - prob_targets).pow(gamma)

    # Compute the focal loss term: - (1 - P(y))^gamma * log(P(y))
    focal_loss_term = -modulating_factor * log_prob_targets

    # Apply alpha weighting (optional)
    if alpha is not None:
        if alpha.device != logits.device:
            alpha = alpha.to(logits.device)
        # Gather the alpha weights corresponding to the target classes
        alpha_weights = alpha.gather(0, targets)
        focal_loss_term = alpha_weights * focal_loss_term

    # Apply reduction
    if reduction == 'mean':
        loss = focal_loss_term.mean()
    elif reduction == 'sum':
        loss = focal_loss_term.sum()
    elif reduction == 'none':
        loss = focal_loss_term
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}. Choose 'none', 'mean', or 'sum'.")

    return loss
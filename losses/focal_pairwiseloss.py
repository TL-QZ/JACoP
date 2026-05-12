import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FocalPairwiseLoss(nn.Module):
    """Focal loss implementation for pairwise potential learning.
    
    Addresses the severe class imbalance in pairwise trajectory prediction
    where we have K²-1 negative examples vs 1 positive example per edge.
    """
    
    def __init__(self, num_modes: int, alpha: float = 0.25, gamma: float = 2.0, 
                 reduction: str = 'mean', smooth_eps: float = 1e-7):
        """
        Args:
            num_modes: Number of trajectory modes (K)
            alpha: Weighting factor for rare class (positive pairs)
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: Loss reduction method ('mean', 'sum', 'none')
            smooth_eps: Small epsilon for numerical stability
        """
        super().__init__()
        self.num_modes = num_modes
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.smooth_eps = smooth_eps
        
    def forward(self, pairwise_potential: torch.Tensor, 
                edge_index: torch.Tensor, 
                best_pred_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for pairwise potentials.
        
        Args:
            pairwise_potential: [E, K, K] pairwise potentials
            edge_index: [2, E] edge indices
            best_pred_idx: [N] best prediction indices for each agent
            
        Returns:
            Focal loss value
        """
        if pairwise_potential.shape[0] == 0:  # No edges
            return torch.tensor(0.0, device=pairwise_potential.device, requires_grad=True)
            
        E, K, _ = pairwise_potential.shape
        device = pairwise_potential.device
        
        # Create ground truth labels
        source_idx = edge_index[0]
        target_idx = edge_index[1]
        
        # Ground truth: one-hot encoding for the correct (source_mode, target_mode) pair
        gt_labels = torch.zeros(E, K * K, device=device)
        gt_flat_idx = best_pred_idx[source_idx] * K + best_pred_idx[target_idx]
        gt_labels[torch.arange(E, device=device), gt_flat_idx] = 1.0
        
        # Convert potentials to probabilities
        logits = pairwise_potential.reshape(E, K * K)
        probs = F.softmax(logits, dim=-1)
        probs = torch.clamp(probs, self.smooth_eps, 1 - self.smooth_eps)  # Numerical stability
        
        # Compute focal loss
        log_probs = torch.log(probs)
        
        # Focal weight: (1 - p_t)^gamma
        pt = torch.sum(gt_labels * probs, dim=-1)  # Probability of true class
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting for class balance
        alpha_weight = self.alpha * gt_labels + (1 - self.alpha) * (1 - gt_labels)
        alpha_t = torch.sum(alpha_weight * gt_labels, dim=-1)
        
        # Final focal loss
        ce_loss = -torch.sum(gt_labels * log_probs, dim=-1)
        focal_loss = alpha_t * focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


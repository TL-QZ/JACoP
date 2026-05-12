"""
Social Collision Metric

This module implements social collision detection for trajectory predictions.
For each scene prediction (K), it counts the number of agent trajectories that
collide with any other agent trajectory in the same prediction.
"""

import torch
from torchmetrics import Metric
from typing import Dict, Optional

class SocialCollisionRate(Metric):
    """
    Metric to compute social collision rate for predicted agent trajectories.

    For each scene prediction (K), counts the number of agent trajectories that
    collide with any other trajectory in the same scene prediction.
    """

    def __init__(
        self,
        min_distance: float = 0.2,  # Minimum safe distance between agents
        max_guesses: int = 20,   # Maximum number of predictions to consider
        **kwargs
    ) -> None:
        """
        Initialize the Social Collision Rate metric.

        Args:
            min_distance (float): Minimum safe distance between agent centers
        """
        super(SocialCollisionRate, self).__init__(**kwargs)
        self.add_state('collision_count', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total_agents', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.min_distance = min_distance
        self.max_guesses = max_guesses

    def update(
        self,
        pred: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        return_individual: bool = False,
        **kwargs
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Update the social collision rate metric.

        Args:
            pred (torch.Tensor): Predicted trajectories of shape (N, K, T, 2)
            valid_mask (torch.Tensor, optional): Mask for valid time steps
        """
        N, K, T, _ = pred.shape

        # Apply valid mask if provided
        if valid_mask is not None:
            mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).expand(N, K, T, 2)
            pred = pred * mask_expanded

        per_agent_collision_count = torch.zeros(N, dtype=pred.dtype, device=pred.device)

        # For each scene prediction (K)
        for k in range(K):
            pred_k = pred[:, k]  # (N, T, 2)
            collided = torch.zeros(N, dtype=torch.bool, device=pred.device)

            # Check collisions for each agent pair
            # Compute pairwise distances for all agent pairs at each timestep
            # pred_k: (N, T, 2)
            # Expand for broadcasting: (N, 1, T, 2) - (1, N, T, 2) -> (N, N, T, 2)
            diff = pred_k.unsqueeze(1) - pred_k.unsqueeze(0)  # (N, N, T, 2)
            dist = torch.norm(diff, dim=-1)  # (N, N, T)

            # Ignore self-distances by setting diagonal to a large value
            dist[torch.arange(N), torch.arange(N), :] = float('inf')

            # Check if any timestep is below min_distance for each agent pair
            collision_matrix = (dist < self.min_distance).any(dim=-1)  # (N, N)

            # An agent is collided if any other agent collides with it
            collided = collision_matrix.any(dim=1)  # (N,)


            collided_f = collided.float()
            per_agent_collision_count += collided_f
            self.collision_count += collided_f.sum()
            self.total_agents += N

        if return_individual:
            batch_collision_count = per_agent_collision_count.sum()
            total_agents_batch = torch.tensor(float(N * K), device=pred.device, dtype=pred.dtype)
            return {
                'per_agent_collision_count': per_agent_collision_count,
                'batch_collision_count': batch_collision_count,
                'batch_total_agents': total_agents_batch,
                'batch_rate': batch_collision_count / total_agents_batch.clamp_min(1.0),
            }
        return None

    def compute(self) -> torch.Tensor:
        """
        Compute the social collision rate.

        Returns:
            torch.Tensor: Social collision rate (0-1)
        """
        if self.total_agents == 0:
            return torch.tensor(0.0, device=self.collision_count.device)
        return self.collision_count / self.total_agents

# Implementation of pairwise potential for MRF predictor (may have multiple types of pairwise potential modules for development convenience)

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import MLPLayer

class DistancePotential(nn.Module):
    """Distance-based pairwise potential
    
    Args:
        type (str): Type of distance potential ("cosine", "euclidean", etc.)"""
    def __init__(self, type: str = "cosine", order: int = 2) -> None:
        super().__init__()
        self.type = type
        self.order = order

    def forward(self,
                source_feat: torch.Tensor,
                target_feat: torch.Tensor,
                ) -> torch.Tensor:
        r"""Calculate the pairwise potential between two agents based on distance
        Args:
            source_feat (torch.Tensor): Feature of the source agent, shape (K,E,D_in)
            target_feat (torch.Tensor): Feature of the target agent, shape (K,E,D_in)
        Returns:
            torch.Tensor: Pairwise potential between two agents, shape (K,K,E)
        """
        K, E, D_in = source_feat.size()
        source_feat = source_feat.unsqueeze(1).expand(K, K, E, D_in)
        target_feat = target_feat.unsqueeze(0).expand(K, K, E, D_in)
        
        if self.type == "cosine":
            pair_potential = F.cosine_similarity(source_feat, target_feat, dim=-1)  # (K,K,E)
        elif self.type == "euclidean":
            pair_potential = -torch.norm(source_feat - target_feat, p=self.order, dim=-1)  # (K,K,E)
        else:
            raise ValueError(f"Unknown distance type: {self.type}")
        
        return pair_potential

class MLPPotential(nn.Module):
    """Vanilla MLP potential for pairwise interaction

    Args:
        nn (_type_): _description_
    """

    def __init__(self,
                 input_dim:int=64*2,
                 hidden_dim:int=64,
                 ) -> None:
        super(MLPPotential, self).__init__()
        self.nn = nn.Sequential(
            MLPLayer(input_dim, hidden_dim, hidden_dim),
            MLPLayer(hidden_dim, hidden_dim, 1),
        )


    def forward(self,
                source_feat:torch.Tensor,
                target_feat:torch.Tensor,
                ) -> torch.Tensor:
        r"""Calculate the pairwise potential between two agents
        Args:
            source_feat (torch.Tensor): Feature of the source agent, shape (K,E,D_in)
            target_feat (torch.Tensor): Feature of the target agent, shape (K,E,D_in)
        Returns:
            torch.Tensor: Pairwise potential between two agents, shape (K,K,E)
        """
        K, E, D_in = source_feat.size()
        source_feat = source_feat.unsqueeze(1).expand(K, K, E, D_in)
        target_feat = target_feat.unsqueeze(0).expand(K, K, E, D_in)
        pair_feat = torch.cat([source_feat, target_feat], dim=-1)  # (K,K,E,2*D_in)
        pair_potential = self.nn(pair_feat).squeeze(-1)  # (K,K,E)
        return pair_potential
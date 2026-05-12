"""
Environmental Collision Detection for ETH-UCY Dataset

This module implements environmental collision detection using binary occupancy maps
and homography transformations to convert trajectory coordinates from world space
to image space for collision checking.
"""

import os
import cv2
import numpy as np
import torch
from torchmetrics import Metric
from typing import Optional, Dict, Tuple
from pathlib import Path

import matplotlib.pyplot as plt

class EnvironmentalCollisionRate(Metric):
    """
    Metric to compute environmental collision rate for predicted trajectories.
    
    This metric uses binary occupancy maps and homography matrices to detect
    collisions with static obstacles in the environment.
    """
    
    def __init__(
        self,
        dataset_root: str,
        scene_name: str,
        max_guesses: int = 6,
        **kwargs
    ) -> None:
        """
        Initialize the Environmental Collision Rate metric.
        
        Args:
            dataset_root (str): Path to the dataset root directory
            scene_name (str): Name of the scene (eth, hotel, univ, zara1, zara2)
            max_guesses (int): Maximum number of trajectory guesses to evaluate
            agent_radius (float): Radius of the agent in meters for collision detection
        """
        super().__init__(**kwargs)
        
        self.add_state('collision_count', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total_count', default=torch.tensor(0), dist_reduce_fx='sum')
        
        self.max_guesses = max_guesses
        self.scene_name = scene_name
        
        # Load binary map and homography matrix
        self.binary_map, self.homography_matrix = self._load_map_data(dataset_root, scene_name)
        
    def _load_map_data(self, dataset_root: str, scene_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load binary occupancy map and homography matrix for the scene.
        
        Args:
            dataset_root (str): Path to the dataset root directory
                                homography matrix must be image to world coordinate
            scene_name (str): Name of the scene
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Binary map and homography matrix
        """
        # Load binary map
        map_path = os.path.join(dataset_root, 'maps', 'maps', f'{scene_name}.png')
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Binary map not found at {map_path}")
            
        # Load as grayscale and convert to binary (0: obstacle, 1: free space)
        if scene_name in ['eth', 'hotel']:
            binary_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE).T 
        else:
            binary_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)

        # Load homography matrix
        homo_path = os.path.join(dataset_root, 'maps', 'homo_mats', f'{scene_name}_H.txt')
        if not os.path.exists(homo_path):
            raise FileNotFoundError(f"Homography matrix not found at {homo_path}")

        homography_matrix = np.linalg.inv(np.loadtxt(homo_path)) # make homography matrix from world to image coordinates

        return binary_map, homography_matrix
    
    def _world_to_image(self, world_coords: torch.Tensor) -> torch.Tensor:
        """
        Convert world coordinates to image coordinates using homography.
        
        Args:
            world_coords (torch.Tensor): World coordinates of shape (..., 2)
            
        Returns:
            torch.Tensor: Image coordinates of shape (..., 2)
        """
        original_shape = world_coords.shape
        world_coords_flat = world_coords.reshape(-1, 2)
        
        # Convert to homogeneous coordinates
        ones = torch.ones(world_coords_flat.shape[0], 1, device=world_coords.device)
        world_homo = torch.cat([world_coords_flat, ones], dim=1)  # Shape: (N, 3)
        
        # Apply homography transformation
        H = torch.tensor(self.homography_matrix, dtype=world_coords.dtype, device=world_coords.device)
        image_homo = torch.matmul(world_homo, H.T)  # Shape: (N, 3)
        
        # Convert back to Cartesian coordinates
        image_coords = image_homo[:, :2] / image_homo[:, 2:3]
        
        return image_coords.reshape(original_shape)
    
    def _check_collision(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Check for environmental collisions at given positions.
        
        Args:
            positions (torch.Tensor): Positions to check of shape (..., 2)
            
        Returns:
            torch.Tensor: Boolean tensor indicating collisions
        """
        # Convert world coordinates to image coordinates
        image_coords = self._world_to_image(positions)
        
        # Convert to integer pixel coordinates
        pixel_coords = image_coords.round().long()
        
        # Get map dimensions
        h, w = self.binary_map.shape
        
        # Check bounds
        valid_x = (pixel_coords[..., 0] >= 0) & (pixel_coords[..., 0] < w)
        valid_y = (pixel_coords[..., 1] >= 0) & (pixel_coords[..., 1] < h)
        valid_coords = valid_x & valid_y
        
        # Initialize collision tensor
        collisions = torch.zeros_like(valid_coords, dtype=torch.bool)
        
        # Check collisions for valid coordinates
        if valid_coords.any():
            valid_pixel_coords = pixel_coords[valid_coords]
            map_tensor = torch.tensor(self.binary_map, device=positions.device)
            
            # Check if positions are in obstacle regions (binary_map value = 0)
            pixel_values = map_tensor[valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]]
            valid_collisions = pixel_values == 0
            
            collisions[valid_coords] = valid_collisions
        
        # Consider out-of-bounds as collisions
        collisions |= ~valid_coords
        
        return collisions
    
    def _check_trajectory_collision(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Check if a trajectory has any environmental collisions.
        
        Args:
            trajectory (torch.Tensor): Trajectory of shape (T, 2)
            
        Returns:
            torch.Tensor: Boolean indicating if trajectory has collision
        """
        # Check collision at each time step
        step_collisions = self._check_collision(trajectory)  # Shape: (T,)
        
        # A trajectory has collision if any step has collision
        return step_collisions.any()
    
    def update(
        self,
        pred: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        return_individual: bool = False,
        **kwargs
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Update the environmental collision rate metric.
        
        Args:
            pred (torch.Tensor): Predicted trajectories of shape (N, K, T, 2)
            prob (torch.Tensor, optional): Probabilities for each prediction
            valid_mask (torch.Tensor, optional): Mask for valid time steps
        """
        N, K, T, _ = pred.shape
        
        # Select top-k predictions based on probability
        if prob is not None:
            _, top_indices = torch.topk(prob, min(self.max_guesses, K), dim=-1)
            pred_topk = torch.gather(
                pred, 1, 
                top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, 2)
            )
        else:
            pred_topk = pred[:, :self.max_guesses]
        
        # Apply valid mask if provided
        if valid_mask is not None:
            # Expand mask to match prediction dimensions
            mask_expanded = valid_mask.unsqueeze(1).expand(-1, pred_topk.shape[1], -1)
            # Set invalid positions to a safe location (no collision)
            pred_topk = pred_topk * mask_expanded.unsqueeze(-1)
        
        # Check collisions for each agent and each prediction
        collision_counts = []
        for n in range(N):
            agent_collisions = []
            for k in range(pred_topk.shape[1]):
                trajectory = pred_topk[n, k]  # Shape: (T, 2)
                has_collision = self._check_trajectory_collision(trajectory)
                agent_collisions.append(has_collision)
            
            # For each agent, take the minimum collision rate across all predictions
            # (i.e., if any prediction is collision-free, the agent is considered safe)
            agent_collision_count = torch.stack(agent_collisions).sum()
            collision_counts.append(agent_collision_count)

        # Update statistics
        per_agent_collision_count = torch.stack(collision_counts).to(pred.dtype)
        collision_count = per_agent_collision_count.sum()
        self.collision_count += collision_count
        self.total_count += N*self.max_guesses

        if return_individual:
            total_count_batch = torch.tensor(float(N * self.max_guesses), device=pred.device, dtype=pred.dtype)
            return {
                'per_agent_collision_count': per_agent_collision_count,
                'batch_collision_count': collision_count,
                'batch_total_count': total_count_batch,
                'batch_rate': collision_count / total_count_batch.clamp_min(1.0),
            }
        return None
    
    def compute(self) -> torch.Tensor:
        """
        Compute the environmental collision rate.
        
        Returns:
            torch.Tensor: Environmental collision rate (0-1)
        """
        if self.total_count == 0:
            return torch.tensor(0.0, device=self.collision_count.device)
        return self.collision_count / self.total_count


def load_scene_collision_detector(dataset_root: str, scene_name: str) -> EnvironmentalCollisionRate:
    """
    Factory function to create an environmental collision detector for a specific scene.
    
    Args:
        dataset_root (str): Path to the dataset root directory
        scene_name (str): Name of the scene (eth, hotel, univ, zara1, zara2)
        
    Returns:
        EnvironmentalCollisionRate: Configured collision detector
    """
    return EnvironmentalCollisionRate(
        dataset_root=dataset_root,
        scene_name=scene_name,
        max_guesses=6,
    )

class EnvironmentalCollisionRate_SDD(Metric): 
    """
    # not the best for SDD, since certain instances spawn in non-navigable areas (inside building) since these agent might be interpolataed in SDD
    # or the segment map is not perfect 
    Metric to compute environmental collision rate for predicted trajectories.
    
    This metric uses binary occupancy maps and homography matrices to detect
    collisions with static obstacles in the environment.
    """
    
    def __init__(
        self,
        dataset_root: str,
        scene_name: str,
        max_guesses: int = 6,
        **kwargs
    ) -> None:
        """
        Initialize the Environmental Collision Rate metric.
        
        Args:
            dataset_root (str): Path to the dataset root directory
            scene_name (str): Name of the scene (eth, hotel, univ, zara1, zara2)
            max_guesses (int): Maximum number of trajectory guesses to evaluate
            agent_radius (float): Radius of the agent in meters for collision detection
        """
        super().__init__(**kwargs)
        
        self.add_state('collision_count', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total_count', default=torch.tensor(0), dist_reduce_fx='sum')
        
        self.max_guesses = max_guesses
        self.scene_name = scene_name
        
        # Load binary map and homography matrix
        self.maps = {}
        self._load_map_data(dataset_root)
        
    def _load_map_data(self, dataset_root: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load binary occupancy map and homography matrix for the scene.
        
        Args:
            dataset_root (str): Path to the dataset root directory
                                homography matrix must be image to world coordinate
            scene_name (str): Name of the scene
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Binary map and homography matrix
        """
        # Load binary map
        map_path = os.path.join(dataset_root, 'semantic_maps')
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Binary map not found at {map_path}")
        map_files = os.listdir(map_path)
        for file in map_files:
            if file.endswith('.png'):
                scene_name = file.split('.png')[0].split('_')
                scene_name = '_'.join(scene_name[:-1]) # file name example: 'bookstore_0_mask.png'
                scene_path = os.path.join(map_path, file)
                seg_image = plt.imread(scene_path)*255
                target_class = 3 # hard-coded since label 3 is solid structure not navigable
                if scene_name == 'coupa_3':
                    target_class = 2 # special case for coupa_3 where label 2 is solid structure not navigable
                scene_map = 1 - (seg_image == target_class).astype(np.uint8)
                # g2l_homo = np.eye(3) # no need for convert
                self.maps[scene_name] = scene_map
    

    def _check_collision(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Check for environmental collisions at given positions.
        
        Args:
            positions (torch.Tensor): Positions to check of shape (..., 2)
            
        Returns:
            torch.Tensor: Boolean tensor indicating collisions
        """
        
        # Convert to integer pixel coordinates
        pixel_coords = positions.round().long()
        
        # Get map dimensions
        h, w = self.binary_map.shape
        
        # Check bounds
        valid_x = (pixel_coords[..., 0] >= 0) & (pixel_coords[..., 0] < w)
        valid_y = (pixel_coords[..., 1] >= 0) & (pixel_coords[..., 1] < h)
        valid_coords = valid_x & valid_y
        
        # Initialize collision tensor
        collisions = torch.zeros_like(valid_coords, dtype=torch.bool)
        
        # Check collisions for valid coordinates
        if valid_coords.any():
            valid_pixel_coords = pixel_coords[valid_coords]
            map_tensor = torch.tensor(self.binary_map, device=positions.device)
            
            # Check if positions are in obstacle regions (binary_map value = 0)
            pixel_values = map_tensor[valid_pixel_coords[:, 1], valid_pixel_coords[:, 0]]
            valid_collisions = pixel_values == 0
            
            collisions[valid_coords] = valid_collisions
        
        # Consider out-of-bounds as collisions
        collisions |= ~valid_coords
        
        return collisions
    
    def _check_trajectory_collision(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Check if a trajectory has any environmental collisions.
        
        Args:
            trajectory (torch.Tensor): Trajectory of shape (T, 2)
            
        Returns:
            torch.Tensor: Boolean indicating if trajectory has collision
        """
        # Check collision at each time step
        step_collisions = self._check_collision(trajectory)  # Shape: (T,)
        
        # A trajectory has collision if any step has collision
        return step_collisions.any()
    
    def update(
        self,
        pred: torch.Tensor,
        prob: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        scene_name: Optional[str] = None,
        return_individual: bool = False,
        **kwargs
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Update the environmental collision rate metric.
        
        Args:
            pred (torch.Tensor): Predicted trajectories of shape (N, K, T, 2)
            prob (torch.Tensor, optional): Probabilities for each prediction
            valid_mask (torch.Tensor, optional): Mask for valid time steps
        """
        N, K, T, _ = pred.shape
        if not scene_name:
            raise ValueError("scene_name must be provided for SDD collision detection.")
        scene_name = scene_name.split('_')[:-1]
        scene_name = '_'.join(scene_name)
        self.binary_map = self.maps[scene_name]
        
        # Select top-k predictions based on probability
        if prob is not None:
            _, top_indices = torch.topk(prob, min(self.max_guesses, K), dim=-1)
            pred_topk = torch.gather(
                pred, 1, 
                top_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, T, 2)
            )
        else:
            pred_topk = pred[:, :self.max_guesses]
        
        # Apply valid mask if provided
        if valid_mask is not None:
            # Expand mask to match prediction dimensions
            mask_expanded = valid_mask.unsqueeze(1).expand(-1, pred_topk.shape[1], -1)
            # Set invalid positions to a safe location (no collision)
            pred_topk = pred_topk * mask_expanded.unsqueeze(-1)
        
        # Check collisions for each agent and each prediction
        collision_counts = []
        for n in range(N):
            agent_collisions = []
            for k in range(pred_topk.shape[1]):
                trajectory = pred_topk[n, k]  # Shape: (T, 2)
                has_collision = self._check_trajectory_collision(trajectory)
                agent_collisions.append(has_collision)
            
            # For each agent, take the minimum collision rate across all predictions
            # (i.e., if any prediction is collision-free, the agent is considered safe)
            agent_collision_count = torch.stack(agent_collisions).sum()
            collision_counts.append(agent_collision_count)

        # Update statistics
        per_agent_collision_count = torch.stack(collision_counts).to(pred.dtype)
        collision_count = per_agent_collision_count.sum()
        self.collision_count += collision_count
        self.total_count += N*self.max_guesses

        if return_individual:
            total_count_batch = torch.tensor(float(N * self.max_guesses), device=pred.device, dtype=pred.dtype)
            return {
                'per_agent_collision_count': per_agent_collision_count,
                'batch_collision_count': collision_count,
                'batch_total_count': total_count_batch,
                'batch_rate': collision_count / total_count_batch.clamp_min(1.0),
            }
        return None
    
    def compute(self) -> torch.Tensor:
        """
        Compute the environmental collision rate.
        
        Returns:
            torch.Tensor: Environmental collision rate (0-1)
        """
        if self.total_count == 0:
            return torch.tensor(0.0, device=self.collision_count.device)
        return self.collision_count / self.total_count
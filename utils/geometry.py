# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import torch


def angle_between_2d_vectors(
        ctr_vector: torch.Tensor,
        nbr_vector: torch.Tensor) -> torch.Tensor:
    return torch.atan2(ctr_vector[..., 0] * nbr_vector[..., 1] - ctr_vector[..., 1] * nbr_vector[..., 0],
                       (ctr_vector[..., :2] * nbr_vector[..., :2]).sum(dim=-1))


def angle_between_3d_vectors(
        ctr_vector: torch.Tensor,
        nbr_vector: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.cross(ctr_vector, nbr_vector, dim=-1).norm(p=2, dim=-1),
                       (ctr_vector * nbr_vector).sum(dim=-1))


def side_to_directed_lineseg(
        query_point: torch.Tensor,
        start_point: torch.Tensor,
        end_point: torch.Tensor) -> str:
    cond = ((end_point[0] - start_point[0]) * (query_point[1] - start_point[1]) -
            (end_point[1] - start_point[1]) * (query_point[0] - start_point[0]))
    if cond > 0:
        return 'LEFT'
    elif cond < 0:
        return 'RIGHT'
    else:
        return 'CENTER'


def wrap_angle(
        angle: torch.Tensor,
        min_val: float = -math.pi,
        max_val: float = math.pi) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


def polar_to_cartesian(polar:torch.Tensor, r_first=True) -> torch.Tensor:
    """Transform polar coordinates to cartesian coordinates.

    Args:
        polar (torch.Tensor): B x 2 or B x N x 2 or more dimension, where the last dimension is (r, theta).
    Returns:
        torch.Tensor: samne as original shape but the last dimension is (x, y).
    """
    if r_first:
        r, theta = polar[..., 0], polar[..., 1]
    else:
        theta, r = polar[..., 0], polar[..., 1]
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.stack((x, y), dim=-1)

def cartesian_to_polar(cartesian:torch.Tensor) -> torch.Tensor:
    """Transform cartesian coordinates to polar coordinates.

    Args:
        cartesian (torch.Tensor): B x 2 or B x N x 2 or more dimension, where the last dimension is (x, y).
    Returns:
        torch.Tensor: samne as original shape but the last dimension is (r, theta).
    """
    polar = torch.zeros_like(cartesian)
    bearing = torch.atan2(cartesian[..., 1], cartesian[..., 0])
    #wrap_angle angle to [-pi, pi]
    bearing = torch.where(bearing > math.pi, bearing - 2 * math.pi, bearing)
    bearing = torch.where(bearing < -math.pi, bearing + 2 * math.pi, bearing)
    polar[..., 0] = bearing
    dist = torch.norm(cartesian, p=2, dim=-1)
    polar[..., 1] = dist
    return polar
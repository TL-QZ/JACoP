from ast import arg
import math
import os
import pickle
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
import matplotlib.pyplot as plt

from tqdm import tqdm
import cv2
from skimage import measure
from shapely.geometry import Point, Polygon, LineString
from shapely import contains_xy

import potrace
from PIL import Image
import io
import base64

from transforms.normalizer import TrajNorm

from .eth_ucy_utils import (
    set_edge_to_one,
    raster_to_lines,
    local_to_global_linestrings,
    vectorize_binary_image,
    local_to_global_polygon,
    DirectionalRangeMapComputer,
    SceneMap
)

class SDDDataset(Dataset):
    """Dataset class for Stanford Drone Dataset (SDD).
    
    Args:
        root (string): the root folder of the dataset. If you've downloaded the raw .tar file, placing it in the root
            folder will skip downloading automatically.
        split (string): specify the split of the dataset: `"train"` | `"val"` | `"test"`.
        raw_dir (string, optional): optionally specify the directory of the raw data. By default, the raw directory is
            path/to/root/split/raw/. If specified, the path of the raw log is path/to/raw_dir/log_id. If all logs
            exist in the raw directory, file downloading/extraction will be skipped. (default: None)
        processed_dir (string, optional): optionally specify the directory of the processed data. By default, the
            processed directory is path/to/root/split/processed/. If specified, the path of the processed .pkl files is
            path/to/processed_dir/*.pkl. If all .pkl files exist in the processed directory, file downloading/extraction
            and data preprocessing will be skipped. (default: None)
        transform (callable, optional): a function/transform that takes in an :obj:`torch_geometric.data.Data` object
            and returns a transformed version. The data object will be transformed before every access. (default: None)
        dim (int, Optional): 2D or 3D data. (default: 2)
        num_historical_steps (int, Optional): the number of historical time steps. (default: 8)
        num_future_steps (int, Optional): the number of future time steps. (default: 12)
        predict_unseen_agents (boolean, Optional): if False, filter out agents that are unseen during the historical
            time steps. (default: False)
    """

    def __init__(self,
                 root: str,
                 split: str,
                 raw_dir: Optional[str] = None,
                 processed_dir: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 dim: int = 2,
                 num_historical_steps: int = 8,
                 num_future_steps: int = 12,
                 predict_unseen_agents: bool = False,
                 skip: int = 1,
                 delim:str = '\t',) -> None:
        root = os.path.expanduser(os.path.normpath(root))
        print(os.curdir)
        if not os.path.isdir(root):
            raise ValueError(f'{root} is not a valid directory')
        if split not in ('train', 'test', 'val'):
            raise ValueError(f'{split} is not a valid split')
        
        self.split = split
        self.map_polygons = {}
        self.binary_maps = {}

        if raw_dir is None:
            raw_dir = os.path.join(root, split)
            self._raw_dir = raw_dir
            if os.path.isdir(self._raw_dir):
                self._raw_file_names = [name for name in os.listdir(self._raw_dir)]
            else:
                raise ValueError(f'{raw_dir} is not a valid directory, please download the raw data first')
        else:
            raw_dir = os.path.expanduser(os.path.normpath(raw_dir))
            self._raw_dir = raw_dir
            if os.path.isdir(self._raw_dir):
                self._raw_file_names = [name for name in os.listdir(self._raw_dir)]
            else:
                raise ValueError(f'{raw_dir} is not a valid directory, please download the raw data first')

        if processed_dir is None:
            processed_dir = os.path.join(root, 'processed', split)
            self._processed_dir = processed_dir
            if os.path.isdir(self._processed_dir):
                self._processed_file_names = [name for name in os.listdir(self._processed_dir) if
                                              os.path.isfile(os.path.join(self._processed_dir, name)) and
                                              name.endswith(('pkl', 'pickle'))]
            else:
                self._processed_file_names = []
        else:
            processed_dir = os.path.expanduser(os.path.normpath(processed_dir))
            self._processed_dir = processed_dir
            if os.path.isdir(self._processed_dir):
                self._processed_file_names = [name for name in os.listdir(self._processed_dir) if
                                              os.path.isfile(os.path.join(self._processed_dir, name)) and
                                              name.endswith(('pkl', 'pickle'))]
            else:
                self._processed_file_names = []
        
        # add common prototype/anchor trajectory dir
        self._prototype_dir = os.path.join(root, 'anchors')
        if not os.path.isdir(self._prototype_dir) or len(os.listdir(self._prototype_dir)) == 0:
            raise FileNotFoundError(f'{self._prototype_dir} is not a valid directory or empty')

        self.dim = dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.num_steps = num_historical_steps + num_future_steps
        self.predict_unseen_agents = predict_unseen_agents

        self._agent_types = ['pedestrian']
        self._agent_categories = ['TRACK_FRAGMENT', 'SCORED_TRACK']
        #load common prototype/anchor trajectory first before processing
        self.normailizer = TrajNorm(ori=True, rot=True, sca=False)
        self._load_prototypes()
        super(SDDDataset, self).__init__(root=root, transform=transform, pre_transform=None, pre_filter=None)
        self.scenes = self._load_scenes()
        self._num_samples = len(self.scenes)

    @property
    def raw_dir(self) -> str:
        return self._raw_dir

    @property
    def processed_dir(self) -> str:
        return self._processed_dir

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return self._raw_file_names

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return self._processed_file_names
    
    def len(self) -> int:
        return self._num_samples
    
    def get(self, idx: int) -> HeteroData:
        curr_scene = self.scenes[idx]
        return HeteroData(curr_scene)
    
    def _compute_prototype_compliancy_scores(self, prototypes: torch.Tensor, city: str, map_root: str) -> torch.Tensor:
        """
        given prototype (global coordinate), compute if it violates or is compliant with map constraints
        prototypes: [num_prototypes, num_future_steps, 2]
        city: city name
        map_root: root directory of the maps

        Returns:
        compliant_labels: [num_prototypes], 1 if all future steps are compliant, 0 otherwise, inbetween compliant_frame/total_future_steps
        """
        _,Tf,_ = prototypes.shape
        if city not in self.binary_maps:
            image_path = os.path.join(map_root, f"{city}_mask.png")
            seg_image = plt.imread(image_path)*255
            target_class = 3 # hard-coded since label 3 is solid structure not navigable
            if city == 'coupa_3':
                target_class = 2 # special case for coupa_3 where label 2 is solid structure not navigable
            image = 1 - (seg_image == target_class).astype(np.uint8)
            g2l_homo = np.eye(3) # no need for convert
            self.binary_maps[city] = SceneMap(data=image, w2m=g2l_homo)
        biMap = self.binary_maps[city]
        prototype_navigability = biMap.check_navigability_with_local(prototypes) # [num_prototypes, num_future_steps]
        compliant_score = prototype_navigability.sum(dim=1) / Tf
        return compliant_score
    
    def _get_map_features(self) -> Dict[str, Any]:
        """legacy feature from QCNet"""
        map_data = {
            'map_polygon': {},
            'map_point': {},
            ('map_point', 'to', 'map_polygon'): {},
            ('map_polygon', 'to', 'map_polygon'): {},
        }
        return map_data
    
    def _process_map_polygons(self, city, map_root):
        image_path = os.path.join(map_root, f"{city}_mask.png")
        seg_image = plt.imread(image_path)*255
        target_class = 3 # hard-coded since label 3 is solid structure not navigable
        if city == 'coupa_3':
            target_class = 2 # special case for coupa_3 where label 2 is solid structure not navigable
        image = 1 - (seg_image == target_class).astype(np.uint8)
        g2l_homo = np.eye(3) # no need for convert
        l2g_homo = np.eye(3) # no need for convert

        edge_image = set_edge_to_one(np.zeros_like(image))
        local_linestrings = raster_to_lines((image), simplify_tolerance=1.0) #adjust tolerance here.
        # if city not in ['zara2']:
        local_linestrings.extend(raster_to_lines(edge_image, simplify_tolerance=1.0))
        global_linestrings = local_to_global_linestrings(local_linestrings, l2g_homo)

        paths, shapely_geometries = vectorize_binary_image((1-image.T)*255)
        # if city == 'eth':
        #     h,w = image.shape
        #     boundary_polygon = Polygon.from_bounds(0,0,h,w)
        #     global_polygons = local_to_global_polygon([boundary_polygon, shapely_geometries[1]], l2g_homo)
        #     global_polygons = [global_polygons[0].difference(global_polygons[1])]
        # else:
        #     global_polygons = local_to_global_polygon(shapely_geometries, l2g_homo)
        global_polygons = local_to_global_polygon(shapely_geometries, l2g_homo)

        self.map_polygons[city] = {}
        self.map_polygons[city]['edges'] = global_linestrings
        self.map_polygons[city]['polygons'] = global_polygons
        self.map_polygons[city]['range_mapper'] = DirectionalRangeMapComputer(global_linestrings)
    
    def _process_env_view(self, scene_dict, map_root='data/eth_ucy/maps'):
        """
        Process the environment view for a given scene. (Lidar Style)
        Output: Updated scene_dict with processed environment view.
            scene_dict['agent']['range_map']: Tensor of shape (num_agents, T, 360, 2) dim:1 radian in global coordinate system, dim 2: distance to the obstacle
        """
        city = scene_dict['city']
        if city not in self.map_polygons:
            self._process_map_polygons(city, map_root)
        map_polygons = self.map_polygons[city]['polygons']
        range_mapper = self.map_polygons[city]['range_mapper']
        positions = scene_dict['agent']['position'].numpy()
        headings = scene_dict['agent']['heading'].numpy()
        heading_angles = np.floor(headings * 360/(2*np.pi)) # round to angle
        valid_masks = scene_dict['agent']['valid_mask'].numpy()
        num_agents, T, _ = positions.shape
        agent_range_maps = np.zeros([num_agents, T, 360, 2]) # 360 degree awareness
        
        for a in range(num_agents):
            agent_pos = positions[a]
            alpha = heading_angles[a]
            # alpha_min = alpha - 179 #assuming 360 degree awareness
            # alpha_max = alpha + 180
            alpha_min, alpha_max = np.zeros_like(alpha), np.zeros_like(alpha)+359
            agent_range_map = range_mapper.compute_directional_range_map_vectorized(
                agent_pos, 
                alpha_min, 
                alpha_max, 
                angle_resolution=1.0,  # Fewer rays for clearer visualization
                max_range=250.0 # Adjust max range as needed (view distance 250 pixels)
            )
            # agent_range_map[:,:,0] = (agent_range_map[:,:,0]-alpha[:,None]+180)%360-180    # wrap around -180 to 180
            # convert to polar coordinates
            agent_range_map[:,:,0] = (np.deg2rad(agent_range_map[:,:,0])+ 2*np.pi) % (2*np.pi)  # angle in radians between 0 and 2pi
            # agent_violates = np.where(np.any([contains_xy(polygon, agent_pos)for polygon in map_polygons],axis=0),-1,1)
            # agent_range_map *= agent_violates[:,None,None]
            agent_range_maps[a] = agent_range_map
        agent_range_maps = agent_range_maps * valid_masks[:,:,None,None]
        scene_dict['agent']['range_map'] = torch.from_numpy(agent_range_maps).to(torch.float)
        return scene_dict
    
    def process(self) -> None:
        for file_name in tqdm(self.raw_file_names):
            scene_name = file_name.split('.')[0]
            scene_path = os.path.join(self.raw_dir, file_name)
            scene = self._process_scenes(scene_path)
            with open(os.path.join(self.processed_dir, f'{scene_name}.pkl'), 'wb') as f:
                pickle.dump(scene, f)

    def _process_scene(self, scene_data: torch.Tensor,
                       frames:torch.Tensor,
                       curr_frame_idx: int,
                       city: str) -> Dict[str, Any]:
        peds_in_curr_seq = torch.unique(scene_data[:, 1])
        num_agents = len(peds_in_curr_seq)
        valid_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        current_valid_mask = torch.zeros(num_agents, dtype=torch.bool)
        predict_mask = torch.zeros(num_agents, self.num_steps, dtype=torch.bool)
        agent_ids: List[Optional[str]] = [None] * num_agents
        agent_type = torch.zeros(num_agents, dtype=torch.uint8)
        agent_category = torch.zeros(num_agents, dtype=torch.uint8)
        position = torch.zeros(num_agents, self.num_steps, 2, dtype=torch.float) 
        heading = torch.zeros(num_agents, self.num_steps, dtype=torch.float)
        velocity = torch.zeros(num_agents, self.num_steps, self.dim, dtype=torch.float)
        anchor_label = torch.zeros(num_agents, self.num_prototypes, dtype=torch.float) # one-hot encoding for corrensponding anchor trajectory
        anchor_complaint_score = torch.zeros(num_agents, self.num_prototypes, dtype=torch.float)

        for agent_idx, agent_id in enumerate(peds_in_curr_seq):
            curr_agent_seq = scene_data[scene_data[:, 1] ==
                                                 agent_id, :]
            curr_agent_seq = np.around(curr_agent_seq, decimals=4)
            pad_front = frames.index(curr_agent_seq[0, 0]) - curr_frame_idx
            pad_end = frames.index(curr_agent_seq[-1, 0]) - curr_frame_idx + 1
            include_for_testing = False if pad_end - pad_front != self.num_steps else True
            position[agent_idx, pad_front:pad_end, :] = curr_agent_seq[:, 2:4]
            valid_mask[agent_idx, pad_front:pad_end] = True
            predict_mask[agent_idx, pad_front:pad_end] = True
            agent_type[agent_idx] = self._agent_types.index('pedestrian')
            agent_category[agent_idx] = self._agent_categories.index('SCORED_TRACK') if include_for_testing \
                                        else self._agent_categories.index('TRACK_FRAGMENT')
            if pad_front < self.num_historical_steps and pad_end >= self.num_historical_steps:
                current_valid_mask[agent_idx] = True
            if not current_valid_mask[agent_idx]:
                predict_mask[agent_idx, self.num_historical_steps:] = False
            agent_ids[agent_idx] = agent_id

            motion_vector = position[agent_idx, 1:] - position[agent_idx, :-1]
            heading[agent_idx, 2:] = torch.atan2(motion_vector[:-1, 1], motion_vector[:-1, 0])
            # heading[agent_idx, 0] = heading[agent_idx, 1] # leave the first and second step's heading to be 0
            heading[agent_idx, :] = (heading[agent_idx, :] + 2 * np.pi) % (2 * np.pi) # make sure angle is between 0 and 2pi
            velocity[agent_idx, 1:] = motion_vector/0.4
            # velocity[agent_idx, 0] = velocity[agent_idx, 1] # leave first step's velocity to be 0
            # for agent with full future trajectory, find the corresponding anchor trajectory
            if predict_mask[agent_idx, -1] and current_valid_mask[agent_idx]:
                #  prepare normalization parameters for current trajectory
                self.normailizer.calculate_params(position[agent_idx:agent_idx+1, :self.num_historical_steps])
                norm_future = self.normailizer.normalize(position[agent_idx:agent_idx+1, self.num_historical_steps:])
                label_idx = torch.argmin(torch.norm(self.prototype_trajs[:,-1] - norm_future[:, -1], dim=-1))
                anchor_label[agent_idx, label_idx] = 1
                agent_prototypes = torch.zeros_like(self.prototype_trajs, dtype=torch.float)
                for p_idx in range(self.num_prototypes):
                    agent_prototypes[p_idx] = self.normailizer.denormalize(self.prototype_trajs[p_idx:p_idx+1])
                compliant_scores = self._compute_prototype_compliancy_scores(
                    agent_prototypes, 
                    city=city,
                    map_root='data/SDD/semantic_maps/'
                ) # [num_prototypes]
                anchor_complaint_score[agent_idx] = compliant_scores

        # only include agent that is observed at current frame
        valid_mask = valid_mask[current_valid_mask]
        predict_mask = predict_mask[current_valid_mask]
        agent_ids = torch.tensor([[agent_ids[i]]for i in range(num_agents) if current_valid_mask[i]])
        agent_type = agent_type[current_valid_mask]
        agent_category = agent_category[current_valid_mask]
        position = position[current_valid_mask]
        velocity = velocity[current_valid_mask]
        heading = heading[current_valid_mask]
        anchor_label = anchor_label[current_valid_mask]
        anchor_complaint_score = anchor_complaint_score[current_valid_mask]
        num_agents = len(agent_ids)

        num_scored_agent = torch.sum(agent_category == self._agent_categories.index('SCORED_TRACK'))
        if num_scored_agent >= 1:
            pass
        else:
            if self.split == 'test': # only include if more than one scored agent
                num_agents = 0
            else:
                if num_scored_agent == 0: # only include if at least one scored agent
                    num_agents = 0
        
        av_idx = [-1]
        curr_scene = {
        'num_nodes': num_agents,
        'av_index': av_idx,
        'valid_mask': valid_mask,
        'predict_mask': predict_mask,
        'id': agent_id,
        'type': agent_type,
        'category': agent_category,
        'position': position,
        'velocity': velocity,
        'heading': heading,
        'anchor_label': anchor_label,
        'anchor_complaint_score': anchor_complaint_score,
        }
        return curr_scene



    def _process_scenes(self, scene_path: str) -> Dict[str, Any]:
        
        with open(scene_path, 'rb') as f:
            data = pickle.load(f)
        
        data['frame'] = data['frame'] // 12  # need to do this to convert to index different by 1
        scenes = []
        for scene_id, scene_data in tqdm(data.groupby('sceneId')):
            if scene_id == 'nexus_2' or scene_id == 'hyang_4':
                # skip incomplete scenes
                continue
            # select columns: frame, trackID, x,y
            scene_data = scene_data[['frame', 'trackId', 'x', 'y']]
            num_frames = scene_data['frame'].nunique()
            frames = torch.from_numpy(scene_data['frame'].unique()).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(torch.from_numpy(scene_data[scene_data['frame'] == frame].to_numpy()))
            num_sequences = int(num_frames - self.num_steps + 1)
            for idx in tqdm(range(num_sequences)):
                curr_seq_data = torch.cat(
                    frame_data[idx: idx +  self.num_steps], dim=0)
                scene_dict = {}
                scene_dict['scenario_id'] = f'{scene_id}_{idx}'
                scene_dict['city'] = scene_id
                scene_dict['agent'] = self._process_scene(curr_seq_data, frames, idx, scene_dict['city'])
                if scene_dict['agent']['num_nodes'] == 0:
                    continue
                scene_dict.update(self._get_map_features())
                scene_dict = self._process_env_view(scene_dict, map_root='data/SDD/semantic_maps/')
                scenes.append(scene_dict)
        return scenes

        
    def _load_prototypes(self) -> None:
        self._prototypes = {}
        file_name = os.listdir(self._prototype_dir)[0]
        self.prototype_trajs = torch.from_numpy(np.load(os.path.join(self._prototype_dir, file_name)))
        self.prototype_trajs = self.prototype_trajs.float() # (num_prototypes, num_fut_steps, 2)
        self.num_prototypes = self.prototype_trajs.size(0)

    def _load_scenes(self) -> List[Dict[str, Any]]:
        scenes = []
        for file_name in self.processed_file_names:
            scene_path = os.path.join(self.processed_dir, file_name)
            with open(scene_path, 'rb') as f:
                scene = pickle.load(f)
                scenes.extend(scene)
        return scenes


if __name__ == '__main__':
    # make sure the sys directory is set to the root of the project
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train', help='data split: train | test')
    args = parser.parse_args()

    dataset = SDDDataset(root='data/SDD', split=args.split)
    print(f'Number of scenes in {args.split} set: {len(dataset)}')
    print('Done!')
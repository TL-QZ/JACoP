import logging
import os
import math
import pandas as pd
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import imageio
from skimage.transform import resize
import pickle5
import pickle

from tqdm import tqdm

logger = logging.getLogger(__name__)


def derivative_of(x, dt=1):

    if x[~np.isnan(x)].shape[-1] < 2:
        return np.zeros_like(x)

    dx = np.full_like(x, np.nan)
    dx[~np.isnan(x)] = np.gradient(x[~np.isnan(x)], dt)
    return dx


def seq_collate(data):
    (obs_seq_list, pred_seq_list,
     map_path, inv_h_t,
     local_map, local_ic, local_homo, scale) = zip(*data)
    scale = scale[0]

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    obs_traj = torch.stack(obs_seq_list, dim=0).permute(2, 0, 1)
    fut_traj = torch.stack(pred_seq_list, dim=0).permute(2, 0, 1)
    seq_start_end = torch.LongTensor(seq_start_end)

    inv_h_t = np.stack(inv_h_t)
    local_ic = np.stack(local_ic)
    local_homo = torch.tensor(np.stack(local_homo)).float().to(obs_traj.device)

    obs_traj_st = obs_traj.clone()
    obs_traj_st[:, :, :2] = (obs_traj_st[:,:,:2] - obs_traj_st[-1, :, :2]) / scale
    obs_traj_st[:, :, 2:] /= scale
    out = [
        obs_traj, fut_traj, obs_traj_st, fut_traj[:,:,2:4] / scale, seq_start_end,
        map_path, inv_h_t,
        local_map, local_ic, local_homo
    ]

    return tuple(out)



def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)




def transform(image, resize):
    im = Image.fromarray(image[0])

    image = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])(im)
    return image

class SDDTrajDataset(Dataset):
    """Dataloader for the Stanford Drone Dataset Trajectory datasets Y-Net format"""

    def __init__(self, data_dir, obs_len=8, pred_len=12, mode='train'):
        """
        Args:
        - data_dir: Directory containing dataset files in the Y-Net format
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - mode: 'train' or 'test'
        """

        super(SDDTrajDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        
        # Load data
        self.data_dir = os.path.join(data_dir, f'{mode}_trajnet.pkl')
        with open(self.data_dir, 'rb') as f:
            self.data = pickle.load(f)
        # Need to change the frame numbers as they are by 12 (0,12,24,...) divide by 12 to get (0,1,2,...)
        # assert all frames are divisible by 12
        assert all(self.data['frame'] % 12 == 0), "Frame numbers are not divisible by 12"
        self.data['frame'] = self.data['frame'] // 12
        # data format: pandas dataframe with columns:['frame', 'trackId', 'x', 'y', 'sceneId, 'metaId]
        # sceneId -- map names
        # Parse data into scenes of 20 frames (8 observed + 12 predicted)
  
        seq_start_end = []
        obs_traj = []
        pred_traj = []
        seq_scene_ids = []
        for scene_id, scene_data in tqdm(self.data.groupby('sceneId')):
            if scene_id == 'nexus_2' or scene_id == 'hyang_4':
                # skip incomplete scenes
                continue
            num_frames = scene_data['frame'].nunique()
            for start_frame in range(0, num_frames - (obs_len + pred_len) + 1):
                end_frame = start_frame + obs_len + pred_len
                seq_data = scene_data[(scene_data['frame'] >= start_frame) & (scene_data['frame'] < end_frame)]
                ped_ids = seq_data['trackId'].unique()
                
                curr_obs_traj = []
                curr_pred_traj = []
                for ped_id in ped_ids:
                    ped_data = seq_data[seq_data['trackId'] == ped_id]
                    if len(ped_data) < obs_len + pred_len:
                        continue
                    ped_data = ped_data.sort_values(by='frame')
                    curr_obs_traj.append(ped_data[['x', 'y']].values[:obs_len])
                    curr_pred_traj.append(ped_data[['x', 'y']].values[obs_len:])
                
                if len(curr_obs_traj) > 0:
                    obs_traj.append(np.array(curr_obs_traj))  # shape (num_peds, obs_len, 2)
                    pred_traj.append(np.array(curr_pred_traj))  # shape (num_peds, pred_len, 2)
                    seq_scene_ids.append(scene_id)
                    seq_start_end.append((len(obs_traj)-1, len(obs_traj)))
        self.obs_traj = torch.from_numpy(np.concatenate(obs_traj, axis=0)).type(torch.float) # shape (total_peds, obs_len, 2)
        self.pred_traj = torch.from_numpy(np.concatenate(pred_traj, axis=0)).type(torch.float)  # shape (total_peds, pred_len, 2)
        self.seq_scene_ids = seq_scene_ids
        self.seq_start_end = seq_start_end
        print(f"Loaded {len(self.obs_traj)} trajectories from {mode} set of SDD.")


class TrajectoryDataset_muse(Dataset):
    """Dataloder for the Trajectory datasets from muse"""

    def __init__(
            self, data_dir, data_split, device='cpu', scale=100
    ):

        super(TrajectoryDataset, self).__init__()

        self.obs_len = 8
        self.pred_len = 12
        self.skip = 1
        self.scale = scale
        self.seq_len = self.obs_len + self.pred_len
        self.delim = ' '
        self.device = device
        if data_split == 'val':
            data_split = 'test'
        self.map_dir = os.path.join(data_dir, 'SDD_semantic_maps', data_split + '_masks')
        self.data_path = os.path.join(data_dir, data_split, data_split + '_trajnet.pkl')
        dt=0.4
        min_ped=0

        self.seq_len = self.obs_len + self.pred_len


        n_state = 6
        num_peds_in_seq = []
        seq_list = []

        obs_frame_num = []
        fut_frame_num = []
        scene_names = []
        local_map_size=[]

        self.stats={}
        self.maps={}
        # for file in os.listdir(self.map_dir):
        #     m = imageio.imread(os.path.join(self.map_dir, file)).astype(float)
        #     self.maps.update({file.split('.')[0]:m})


        with open(self.data_path, 'rb') as f:
            data = pickle5.load(f)

        data = pd.DataFrame(data)
        scenes = data['sceneId'].unique()
        for s in tqdm(scenes):
            # incomplete dataset - trajectories are not aligned with segmentation.
            if ('nexus_2' in s) or ('hyang_4' in s):
                continue
            scene_data = data[data['sceneId'] == s]
            scene_data = scene_data.sort_values(by=['frame', 'trackId'], inplace=False)


            frames = scene_data['frame'].unique().tolist()
            scene_data = np.array(scene_data)
            # map_size = self.maps[s + '_mask'].shape
            # scene_data[:,2] = np.clip(scene_data[:,2], a_min=None, a_max=map_size[1]-1)
            # scene_data[:,3] = np.clip(scene_data[:,3], a_min=None, a_max=map_size[0]-1)


            frame_data = []
            for frame in frames:
                frame_data.append(scene_data[scene_data[:, 0]==frame])

            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / self.skip))

            this_scene_seq = []

            for idx in range(0, num_sequences * self.skip + 1, self.skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len],
                    axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])  # unique agent id

                curr_seq = np.zeros((len(peds_in_curr_seq), n_state, self.seq_len))
                num_peds_considered = 0
                ped_ids = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if (pad_end - pad_front != self.seq_len) or (curr_ped_seq.shape[0] != self.seq_len):
                        continue
                    ped_ids.append(ped_id)
                    # x,y,x',y',x'',y''
                    x = curr_ped_seq[:, 2].astype(float)
                    y = curr_ped_seq[:, 3].astype(float)
                    vx = derivative_of(x, dt)
                    vy = derivative_of(y, dt)
                    ax = derivative_of(vx, dt)
                    ay = derivative_of(vy, dt)

                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = np.stack([x, y, vx, vy, ax, ay])
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    num_peds_in_seq.append(num_peds_considered)
                    seq_list.append(curr_seq[:num_peds_considered])
                    this_scene_seq.append(curr_seq[:num_peds_considered, :2])
                    obs_frame_num.append(np.ones((num_peds_considered, self.obs_len)) * frames[idx:idx + self.obs_len])
                    fut_frame_num.append(
                        np.ones((num_peds_considered, self.pred_len)) * frames[idx + self.obs_len:idx + self.seq_len])
                    scene_names.append([s] * num_peds_considered)


            this_scene_seq = np.concatenate(this_scene_seq)

            per_step_dist = []
            for traj in this_scene_seq:
                traj = traj.transpose(1, 0)
                per_step_dist.append(np.sqrt(((traj[1:] - traj[:-1]) ** 2).sum(1)).sum())
            per_step_dist = np.array(per_step_dist)

            per_step_dist = np.clip(per_step_dist, a_min=240, a_max=None)

            local_map_size.extend(np.round(per_step_dist).astype(int))
            # print( self.maps[s + '_mask'].shape, ': ' ,(per_step_dist).max())

        seq_list = np.concatenate(seq_list, axis=0) # (32686, 2, 16)
        self.obs_frame_num = np.concatenate(obs_frame_num, axis=0)
        self.fut_frame_num = np.concatenate(fut_frame_num, axis=0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.obs_traj = self.obs_traj.permute(0, 2, 1)[ :, :, :2]
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.pred_traj = self.pred_traj.permute(0, 2, 1)[:, :, :2]
        
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        self.map_file_name = np.concatenate(scene_names)
        self.num_seq = len(self.obs_traj)
        self.local_map_size = np.stack(local_map_size)
        self.local_ic = [[]] * self.num_seq
        self.local_homo = [[]] * self.num_seq

        print(self.seq_start_end[-1])

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        pass

if __name__ == "__main__":
    # data_dir = 'data/SDD/'
    # dataset = TrajectoryDataset(data_dir, 'train', device='cpu')
    # print(len(dataset))
    # print(dataset.obs_traj.shape)

    data_dir = 'data/SDD/train/'
    SDDTrajDataset(data_dir)
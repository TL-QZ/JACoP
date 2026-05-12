import os
import math
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import DataLoader
from PIL import Image

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


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non-linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.02, min_ped=1, delim='\t'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non-linear traj when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a sequence
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = sorted(os.listdir(self.data_dir))
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        loss_mask_list = []
        non_linear_ped = []
        frame_list = []
        scene_id = []
        frame_abs_list = []
        self.homography = {}
        self.vector_field = {}
        scene_img_map = {'biwi_eth': 'seq_eth', 'biwi_hotel': 'seq_hotel',
                         'students001': 'students003', 'students003': 'students003', 'uni_examples': 'students003',
                         'crowds_zara01': 'crowds_zara01', 'crowds_zara02': 'crowds_zara02', 'crowds_zara03': 'crowds_zara02'}

        for path in all_files:
            parent_dir, scene_name = os.path.split(path)
            parent_dir, phase = os.path.split(parent_dir)
            parent_dir, dataset_name = os.path.split(parent_dir)
            scene_name, _ = os.path.splitext(scene_name)
            scene_name = scene_name.replace('_' + phase, '')
            # self.vector_field[scene_name] = np.load(os.path.join(parent_dir, "vectorfield", scene_img_map[scene_name] + "_vector_field.npy"))
            
            # if dataset_name in ["eth", "hotel", "univ", "zara1", "zara2"]:
            #     homography_file = os.path.join(parent_dir, "homography", scene_name + "_H.txt")
            #     self.homography[scene_name] = np.loadtxt(homography_file)
            # elif dataset_name in [aa + '2' + bb for aa in ['A', 'B', 'C', 'D', 'E'] for bb in ['A', 'B', 'C', 'D', 'E'] if aa != bb]:
            #     homography_file = os.path.join(parent_dir, "homography", scene_name + "_H.txt")
            #     self.homography[scene_name] = np.loadtxt(homography_file)

            # Load data
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                frame_ids = np.zeros((len(peds_in_curr_seq)))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    if pad_end - pad_front != self.seq_len:
                        continue
                    frame_id = curr_ped_seq[:,0]
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    frame_ids[_idx] = frame_id[0]
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(poly_fit(curr_ped_seq, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    frame_list.extend([frames[idx]] * num_peds_considered)
                    scene_id.extend([scene_name] * num_peds_considered)
                    frame_abs_list.extend(frame_ids[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        self.num_peds_in_seq = np.array(num_peds_in_seq)
        self.frame_list = np.array(frame_list, dtype=np.int32)
        self.scene_id = np.array(scene_id)
        self.frame_abs_list = np.array(frame_abs_list)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float).permute(0, 2, 1)  # NTC
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float).permute(0, 2, 1)  # NTC
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float).gt(0.5)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float).gt(0.5)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        self.frame_list = torch.from_numpy(self.frame_list).type(torch.long)
        self.frame_abs_list = torch.from_numpy(self.frame_abs_list).type(torch.long)
        self.anchor = None
        self.anchor_label = None ## added to anchor classification

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = {"obs_traj": self.obs_traj[start:end],
               "pred_traj": self.pred_traj[start:end],
               "anchor": self.anchor[start:end],
               "anchor_label": self.anchor_label[start:end], ## added to anchor classification
               "non_linear_ped": self.non_linear_ped[start:end],
               "loss_mask": self.loss_mask[start:end],
               "scene_mask": None,
               "seq_start_end": [[0, end - start]],
               "frame": self.frame_list[start:end],
               "scene_id": self.scene_id[start:end]}
        return out

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse

import json
import torch
import numpy as np

from utils import *
from sgan_loader import TrajectoryDataset
from sdd_loader import SDDTrajDataset
from SingularTrajectory.space import SingularSpace
from SingularTrajectory.anchor import AdaptiveAnchor

def main(args):
    # load or setup configuration of dataset, singularspace and anchor classes

    config = get_exp_config(f"prepare/config/{args.dataset}_anchor.json")
    print_arguments(config)

    obs_len, pred_len = config.obs_len, config.pred_len

    # load dataset
    if args.dataset == 'eth':
        data_dir = os.path.join(config.dataset_dir, config.dataset, 'train')
        traj_dataset = TrajectoryDataset(data_dir, obs_len, pred_len, skip=1)
        singular_space = SingularSpace(hyper_params=config,
                                   norm_ori=True, 
                                   norm_rot=True, 
                                   norm_sca=False)
        
    elif args.dataset == 'sdd':
        data_dir = os.path.join(config.dataset_dir, config.dataset)
        traj_dataset = SDDTrajDataset(data_dir)
        singular_space = SingularSpace(hyper_params=config,
                                   norm_ori=True, 
                                   norm_rot=True, 
                                   norm_sca=False)
        
    # load singularspace
    
    
    # load anchor
    anchor = AdaptiveAnchor(hyper_params=config)
    
    # obtain anchors
    obs_traj, pred_traj = traj_dataset.obs_traj, traj_dataset.pred_traj
    space_param = singular_space.parameter_initialization(obs_traj, pred_traj)
    anchor.anchor_initialization(*space_param)
    singular_anchors = anchor.C_anchor.detach()
    euclidean_anchors = singular_space.to_Euclidean_space(C=singular_anchors,
                                                                evec=singular_space.V_pred_trunc)
    print(euclidean_anchors.shape)
    # save anchors
    if args.save_path is not None:
        save_path = args.save_path
    else:
        save_path = os.path.join(config.dataset_dir, config.dataset, 'anchors')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(os.path.join(save_path, 'trajectory_prototypes.npy'), euclidean_anchors.numpy())
    print(f"Anchors saved to {save_path}")

if __name__ == "__main__":
    parse_args = argparse.ArgumentParser()
    parse_args.add_argument("--dataset", type=str, default="sdd", help="dataset name") # [eth, sdd]
    parse_args.add_argument('--save_path', type=str, default=None, help='path to save anchors')
    
    args = parse_args.parse_args()
    print("Preparing anchors for dataset: ", args.dataset)
    main(args)
    print("Done!")
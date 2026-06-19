import argparse
import os
import sys
from copy import deepcopy

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from prepare.SingularTrajectory.anchor import AdaptiveAnchor
from prepare.SingularTrajectory.space import SingularSpace
from prepare.sdd_loader import SDDTrajDataset
from prepare.sgan_loader import TrajectoryDataset
from prepare.utils import DotDict, get_exp_config, print_arguments


ETH_UCY_SPLITS = ('eth', 'hotel', 'univ', 'zara1', 'zara2')


def _load_config(dataset):
    config_path = os.path.join(
        os.path.dirname(__file__),
        'config',
        f'{dataset}_anchor.json',
    )
    return get_exp_config(config_path)


def _copy_config(config):
    return DotDict(deepcopy(dict(config)))


def _build_anchor_config(base_config, num_outputs, num_samples):
    config = _copy_config(base_config)
    if num_outputs is not None and num_samples is not None and num_outputs != num_samples:
        raise ValueError('--num-outputs and --num-samples both control the number of anchors and must match.')

    num_anchors = num_outputs
    if num_anchors is None:
        num_anchors = num_samples
    if num_anchors is None:
        num_anchors = config.num_outputs

    config.num_outputs = num_anchors
    # AdaptiveAnchor uses num_samples as its KMeans cluster count.
    config.num_samples = num_anchors
    return config


def _generate_from_trajectories(config, obs_traj, pred_traj, output_dir, overwrite):
    output_path = os.path.join(output_dir, 'trajectory_prototypes.npy')
    if os.path.exists(output_path) and not overwrite:
        print(f'Skipping existing anchors: {output_path}')
        return output_path

    os.makedirs(output_dir, exist_ok=True)
    print_arguments(config)

    singular_space = SingularSpace(
        hyper_params=config,
        norm_ori=True,
        norm_rot=True,
        norm_sca=False,
    )
    anchor = AdaptiveAnchor(hyper_params=config)

    space_param = singular_space.parameter_initialization(obs_traj, pred_traj)
    anchor.anchor_initialization(*space_param)
    singular_anchors = anchor.C_anchor.detach()
    euclidean_anchors = singular_space.to_Euclidean_space(
        C=singular_anchors,
        evec=singular_space.V_pred_trunc,
    )

    anchors = euclidean_anchors.cpu().numpy()
    np.save(output_path, anchors)
    print(f'Saved anchors to {output_path} with shape {anchors.shape}')
    return output_path


def generate_eth_ucy_anchors(args):
    base_config = _load_config('eth')
    for split in args.splits:
        config = _build_anchor_config(base_config, args.num_outputs, args.num_samples)
        config.dataset_dir = args.data_root
        config.dataset = split

        train_dir = os.path.join(args.data_root, split, 'train')
        output_dir = os.path.join(args.output_root, split, 'anchors')
        print(f'Generating ETH/UCY anchors for split: {split}')
        traj_dataset = TrajectoryDataset(
            train_dir,
            obs_len=config.obs_len,
            pred_len=config.pred_len,
            skip=config.skip,
        )
        _generate_from_trajectories(
            config=config,
            obs_traj=traj_dataset.obs_traj,
            pred_traj=traj_dataset.pred_traj,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )


def generate_sdd_anchors(args):
    base_config = _load_config('sdd')
    config = _build_anchor_config(base_config, args.num_outputs, args.num_samples)
    config.dataset_dir = os.path.join(args.data_root, 'train')
    config.dataset = ''

    print('Generating SDD anchors')
    traj_dataset = SDDTrajDataset(
        config.dataset_dir,
        obs_len=config.obs_len,
        pred_len=config.pred_len,
        mode='train',
    )
    _generate_from_trajectories(
        config=config,
        obs_traj=traj_dataset.obs_traj,
        pred_traj=traj_dataset.pred_traj,
        output_dir=os.path.join(args.output_root, 'anchors'),
        overwrite=args.overwrite,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate JACoP trajectory-prototype anchors.',
    )
    parser.add_argument(
        '--dataset',
        choices=['eth_ucy', 'sdd'],
        required=True,
        help='Dataset family to generate anchors for.',
    )
    parser.add_argument(
        '--data-root',
        default=None,
        help='Dataset root. Defaults to data/eth_ucy for ETH/UCY and data/SDD for SDD.',
    )
    parser.add_argument(
        '--output-root',
        default=None,
        help='Root where anchors are written. Defaults to --data-root.',
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=list(ETH_UCY_SPLITS),
        choices=ETH_UCY_SPLITS,
        help='ETH/UCY leave-one-out splits to process.',
    )
    parser.add_argument(
        '--num-outputs',
        type=int,
        default=None,
        help='Number of trajectory prototypes to generate. Defaults to the config value.',
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Compatibility alias for --num-outputs. If both are set, they must match.',
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Regenerate anchors even if trajectory_prototypes.npy already exists.',
    )
    args = parser.parse_args()

    if args.data_root is None:
        args.data_root = 'data/eth_ucy' if args.dataset == 'eth_ucy' else 'data/SDD'
    if args.output_root is None:
        args.output_root = args.data_root
    return args


def main():
    args = parse_args()
    if args.dataset == 'eth_ucy':
        generate_eth_ucy_anchors(args)
    else:
        generate_sdd_anchors(args)
    print('Done.')


if __name__ == '__main__':
    main()

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
from argparse import ArgumentParser

import torch
import lightning.pytorch as pl
from torch_geometric.loader import DataLoader

from datasets import ETHUCYDataset, SDDDataset
from predictors import MRF
from lightning.pytorch import loggers as pl_loggers


if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--exp_name', type=str, required=False)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--ckpt_path', type=str, required=True)
    parser.add_argument('--collision_thresh', type=float, default=0.2)
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Whether to overwrite sampling config in test time.')
    parser.add_argument('--sample_scales_new', type=int, nargs='+', default=[4,4,8],)
    parser.add_argument('--sampling_mode', type=str, default='reranking', choices=['reranking', 'gibbs',])
    # JACoP_Soft_Anchor sampling arguments
    parser.add_argument('--enable_sampling', action='store_true', default=None,
                        help='Enable sampling from num_hypothesis to num_modes (JACoP_Soft_Anchor only)')
    parser.add_argument('--sample_policy', type=str, default=None, choices=['per_mode', 'mixture'],
                        help='Sampling policy for JACoP_Soft_Anchor (per_mode or mixture)')
    parser.add_argument('--sample_num_modes', type=int, default=None,
                        help='Target number of modes to sample to (JACoP_Soft_Anchor only)')
    parser.add_argument('--sample_temperature', type=float, default=None,
                        help='Temperature for mixture sampling policy (JACoP_Soft_Anchor only)')
    parser.add_argument('--save_output', action='store_true', default=False,
                        help='Whether to save the output predictions to a file.')
    parser.add_argument('--save_path', type=str, default='./outputs/',
                        help='The path to save the output predictions if --save_output is set.')
    parser.add_argument('--log_tensorboard', action='store_true', default=False,
                        help='Whether to log results to TensorBoard.')
    args = parser.parse_args()
    

    checkpoints = torch.load(args.ckpt_path, map_location='cpu')
    hyperparams = checkpoints['hyper_parameters']
    if args.overwrite:
        hyperparams['sampling_mode'] = args.sampling_mode
        hyperparams['save_output'] = args.save_output
        hyperparams['save_path'] = args.save_path
        hyperparams['collision_thresh'] = args.collision_thresh
    model = MRF(**hyperparams)
    model.load_state_dict(checkpoints['state_dict'], strict=False)
    
    tb_logger = False

    test_dataset = ETHUCYDataset(root=args.root, split='test')
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=args.pin_memory, persistent_workers=args.persistent_workers)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices, strategy='ddp', logger=tb_logger)
    trainer.test(model, dataloader)

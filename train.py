from argparse import ArgumentParser

import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch import loggers as pl_loggers

from datamodules import ETHUCYDataModule, SDDDataModule
from predictors.MRF import MRF

if __name__ == '__main__':
    pl.seed_everything(2023, workers=True)

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='MRF', choices=['MRF'])
    parser.add_argument('--exp_name', type=str, default='dev')
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--train_batch_size', type=int, required=True)
    parser.add_argument('--val_batch_size', type=int, required=True)
    parser.add_argument('--test_batch_size', type=int, required=True)
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--persistent_workers', type=bool, default=True)
    parser.add_argument('--train_raw_dir', type=str, default=None)
    parser.add_argument('--val_raw_dir', type=str, default=None)
    parser.add_argument('--test_raw_dir', type=str, default=None)
    parser.add_argument('--train_processed_dir', type=str, default=None)
    parser.add_argument('--val_processed_dir', type=str, default=None)
    parser.add_argument('--test_processed_dir', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, required=True)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--pretrained_ckpts_path', type=str, default=None)
    parser.add_argument('--monitor_metric', type=str, default='Val/minFDE')
    parser.add_argument('--early_stop_patience', type=int, default=0)
    
    MODELS = {
        'MRF': MRF,
    }
    # Parse known args to determine the model
    temp_args, _ = parser.parse_known_args()
    MODELS[temp_args.model].add_model_specific_args(parser)
    args = parser.parse_args()  
    model = MODELS[args.model](**vars(args))
   
    model_checkpoint = ModelCheckpoint(monitor='Val/minFDE', save_top_k=3, mode='min', save_last=True)

    log_dir = f"lightning_logs/{args.model}/{args.exp_name}/"
    exp_name = f"{args.root.split('/')[-1]}"
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=log_dir, name=exp_name)

    # hardcode batch size to 1 for validation and testing
    args.val_batch_size = 1
    args.test_batch_size = 1
    datamodule = {
        'eth_ucy': ETHUCYDataModule,
        'sdd': SDDDataModule,
    }[args.dataset](**vars(args))
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks = [model_checkpoint, lr_monitor]
    if args.early_stop_patience > 0:
        early_stop = EarlyStopping(monitor=args.monitor_metric, patience=args.early_stop_patience, mode='min')
        callbacks.append(early_stop)
    trainer = pl.Trainer(accelerator=args.accelerator, devices=args.devices,
                         strategy=DDPStrategy(find_unused_parameters=True, 
                                              gradient_as_bucket_view=True),
                         callbacks=callbacks, 
                         max_epochs=args.max_epochs,
                         logger=tb_logger,
                         check_val_every_n_epoch=5)
    trainer.fit(model, datamodule)
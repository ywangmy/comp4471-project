import argparse
import json
import os
import random

import numpy as np
import cv2

import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from comp4471 import loss
from comp4471.model import ASRID
from comp4471.train import train_loop
from config import load_config
from dataloader.loader import configure_data

def my_parse_args():
    parser = argparse.ArgumentParser("ASRID")
    parser.add_argument('--comment', type=str)
    parser.add_argument('--config', metavar='CONFIG_FILE', help='path to configuration file', default='conf.json')
    parser.add_argument('--is-distributed', help='is distributed', action='store_true') # action-> having means True
    parser.add_argument('--workers', type=int, default=6, help='number of cpu threads to use')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--root-dir', type=str, default="data/")
    parser.add_argument('--folds-csv-path', type=str, default='folds.csv')
    parser.add_argument('--crops-dir', type=str, default='crops')
    parser.add_argument('--output-dir', type=str, default='weights/')
    parser.add_argument('--logdir', type=str, default='logs')
    return parser.parse_args()

def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        #torch.backends.cudnn.deterministic = True

def main():
    args = my_parse_args()
    config = load_config(args.config)
    setup_seed(config['seed'])

    # Configure data
    sampler_train, loader_train, loader_val = configure_data(args, config)

    # Distributed
    print(f'is_distributed flag={args.is_distributed}')
    if args.is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        # dist.init_process_group(backend="nccl")
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:6666', world_size=1, rank=local_rank)

    # Device
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load classifier into model
    model = ASRID(batch_size=config['optimizer']['batch_size']).to(device)

    # Resume
    lr = config['optimizer']['lr']
    lr_decay = config['optimizer']['schedule']['lr_decay']
    num_epoch = config['optimizer']['schedule']['num_epoch']
    start_epoch = config['optimizer']['schedule']['start_epoch']
    weight_delay= config['optimizer']['weight_decay']

    # Writer
    writer = SummaryWriter(comment = args.comment)
    # - purge_step: redo experiment from step [purge_step]
    # - comment: show the stage/usage of experiments, e.g. LR_0.1_BATCH_16
    # - log_dir: use default(runs/CURRENT_DATETIME_HOSTNAME)
    #writer.add_hparams({'lr': lr, 'bsize': config['optimizer']['batch_size']}, {})

    optimizer1 = torch.optim.Adam([
        {'params': model.efficientNet.parameters(), 'lr': lr*10},
        {'params': model.multiattn_block.parameters()},
        {'params': model.static_block.parameters()}
    ], lr=lr, weight_decay=weight_delay)
    lr_scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.3, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    start_epoch = train_loop(
        model = model, num_epoch=num_epoch, device=device, writer=writer,
        sampler_train=sampler_train, loader_train=loader_train, loader_val=loader_val,
        optimizer=optimizer1, lr_sheduler=lr_scheduler1, 
	loss_func=loss.twoPhaseLoss(phase=1).to(device), eval_func=loss.evalLoss().to(device),
        start_epoch=start_epoch, phase=1,
        start_iter=0) # kwargs

    optimizer2 = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_delay)
    lr_scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=0.3, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    start_epoch = train_loop(
        model = model, num_epoch=num_epoch - num_epoch / 2, device=device, writer=writer,
        sampler_train=sampler_train, loader_train=loader_train, loader_val=loader_val,
        optimizer=optimizer2, lr_sheduler=lr_scheduler2, 
        loss_func=loss.twoPhaseLoss(phase=2).to(device), eval_func=loss.evalLoss().to(device),
        start_epoch=start_epoch, phase=2,
        start_iter=0) # kwargs

if __name__ == '__main__':
    main()

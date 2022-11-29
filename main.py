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

import comp4471
from comp4471.model import ASRID
from config import load_config
from dataloader.loader import configure_data

def my_parse_args():
    parser = argparse.ArgumentParser("ASRID")
    parser.add_argument('--config', metavar='CONFIG_FILE', help='path to configuration file')
    parser.add_argument('--is-distributed', help='is distributed', action='store_true')
    parser.add_argument('--workers', type=int, default=6, help='number of cpu threads to use')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--root-dir', type=str, default="data/")
    parser.add_argument('--folds-csv-path', type=str, default='folds.csv')
    parser.add_argument('--crops-dir', type=str, default='crops')
    parser.add_argument('--output-dir', type=str, default='weights/')
    parser.add_argument('--logdir', type=str, default='logs')
    return parser.parse_args()

def main():
    args = my_parse_args()
    config = load_config(args.config)

    print(f'is_distributed flag={args.is_distributed}')
    
    if args.is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        # dist.init_process_group(backend="nccl")
        dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:6666', world_size=1, rank=local_rank)

    # Configure data
    sampler_train, loader_train, loader_val = configure_data(args, config)

    # Train loop
    start_epoch = 0

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load classifier into model
    model = ASRID(batch_size=config['optimizer']['batch_size']).to(device)

    # Resume
    lr = config['optimizer']['lr']
    num_epoch = config['optimizer']['schedule']['epochs']
    weight_delay= config['optimizer']['weight_decay']

    def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     #torch.backends.cudnn.deterministic = True

    optimizer1 = torch.optim.Adam(params=[model.multiattn_block.parameters(), model.static_block.parameters()], lr=lr, weight_decay=weight_delay)
    comp4471.train.train_loop(model = model, num_epoch=num_epoch, sampler=sampler_train,
            loader_train=loader_train, loader_val=loader_val,
            optimizer=optimizer1,
            loss_func=comp4471.loss.twoPhaseLoss(phase=1).to(device), eval_func=comp4471.loss.evalLoss().to(device),
            device=device,
            phase=1)
    optimizer2 = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_delay)
    comp4471.train.train_loop(model = model, num_epoch=num_epoch - num_epoch / 2, sampler=sampler_train,
            loader_train=loader_train, loader_val=loader_val,
            optimizer=optimizer2,
            loss_func=comp4471.loss.twoPhaseLoss(phase=2).to(device), eval_func=comp4471.loss.evalLoss().to(device),
            device=device,
            phase=2)

if __name__ == '__main__':
    main()

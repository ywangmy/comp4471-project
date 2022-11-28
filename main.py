import argparse
import json
import os
import cv2

import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import comp4471
import config
import dataloader

# python -m main.py --config conf.json
def my_parse_args():
    parser = argparse.ArgumentParser("ASRID")
    parser.add_argument('--config', metavar='CONFIG_FILE', help='path to configuration file')
    parser.add_argument('--workers', type=int, default=6, help='number of cpu threads to use')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--data-dir', type=str, default="data/")
    parser.add_argument('--folds-csv', type=str, default='folds.csv')
    parser.add_argument('--crops-dir', type=str, default='crops')
    parser.add_argument('--output-dir', type=str, default='weights/')
    parser.add_argument('--logdir', type=str, default='logs')
    return parser.parse_args()

def main():
    args = my_parse_args()
    config = config.load_config(args.config)

    # Configure data
    sampler_train, sampler_val, loader_train, loader_val = dataloader.configure_data(args, config)

    # Train loop
    start_epoch = 0

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load classifier into model
    model = comp4471.model.ASRID().to(device)

    # Resume
    lr = config['optimizer']['schedule']['lr']
    num_epochs = config['optimizer']['schedule']['epochs']

    optimizer1 = torch.optim.Adam(params=[model.MAT.parameters(), model.static.parameters()], lr=lr, weight_decay=0)
    comp4471.train.train_loop(model = model, num_epoch=num_epoch, sampler=sampler_train,
            loader_train=loader_train, loader_val=loader_val,
            optimizer=optimizer1,
            loss_func=comp4471.loss.twoPhaseLoss(phase=1).to(device), eval_func=evalLoss().to(device),
            device=device,
            phase=1)
    optimizer2 = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0)
    comp4471.train.train_loop(model = model, num_epoch=num_epoch - num_epoch / 2, sampler=sampler_train,
            loader_train=loader_train, loader_val=loader_val,
            optimizer=optimizer2,
            loss_func=comp4471.loss.twoPhaseLoss(phase=2).to(device), eval_func=evalLoss().to(device),
            device=device,
            phase=2)

if __name__ == '__main__':
    main()

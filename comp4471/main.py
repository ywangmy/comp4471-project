import argparse
import json
import os
import cv2

from torch.utils.tensorboard import SummaryWriter

import torch
from torch.backends import cudnn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist

from config import load_config
from datasets.augment import create_transforms_train, create_transforms_val


def my_parse_args():
    parser = argparse.ArgumentParser("Hello")
    parser.add_argument('--config', metavar='CONFIG_FILE', help='path to configuration file')
    parser.add_argument('--workers', type=int, default=6, help='number of cpu threads to use')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--data-dir', type=str, default="data/")
    parser.add_argument('--folds-csv', type=str, default='folds.csv')
    parser.add_argument('--crops-dir', type=str, default='crops')
    parser.add_argument('--output-dir', type=str, default='weights/')
    parser.add_argument('--logdir', type=str, default='logs')
    return parser.parse_args()

from train import train_epoch, validate_epoch
def train_loop(model, num_epoch, sampler, loader_train, loader_val, optimizer, loss_func, device, start_epoch = 0, val_freq = 1, verbose = True, **kwargs):
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, verbose=True)
    for epoch in range(num_epochs):
        if verbose:
            print(f'epoch {start_epoch + epoch} in progress')
        sampler.set_epoch(epoch)
        sampler.dataset.next_epoch()
        train_epoch(model, loader_train, optimizer, loss_func, device, kwargs)
        lr_scheduler.step()
        if epoch % val_freq == 0:
            validate_epoch(model, loader_val, kwargs)

def configure_data(args, config):
    # Transforms (augmentation, converting img to tensor, etc.)
    trans_train = create_transforms_train(config['size'])
    trans_val = create_transforms_val(config['size'])

    # Dataset
    data_train = DfdcDataset(mode='train',
                             root_dir=None,
                             crops_dir=None,
                             fold=None,
                             folds_csv_path=None,
                             trans=None)
    data_val = DfdcDataset(mode='val',
                           root_dir=None,
                           crops_dir=None,
                           fold=None,
                           folds_csv_path=None,
                           trans=None)

    # Sampler
    sampler_train = torch.utils.data.distributed.DistributedSampler(data_train)
    sampler_val = torch.utils.data.distributed.DistributedSampler(data_val)

    # Loader
    loader_train = DataLoader(data_train,
                              batch_size=config.batch_size,
                              shuffle=train_sampler is None,
                              sampler=sampler_train,
                              pin_memory=False,
                              num_workers=args.workers,
                              )

    loader_val = DataLoader(data_val,
                            batch_size=config.batch_size * 2, # ???
                            sampler=sampler_val,
                            suffle=False,
                            pin_memory=True,
                            num_workers=args.workers,
                            )
    return sampler_train, sampler_val, loader_train, loader_val

from model import ASRID
from loss import twoPhaseLoss
def main():
    args = my_parse_args()
    config = load_config(args.config)

    # Configure data
    sampler_train, sampler_val, loader_train, loader_val = configure_data(args, config)

    # Train loop
    start_epoch = 0

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load classifier into model
    model = ASRID().to(device)

    # Resume
    lr = config['optimizer']['schedule']['lr']
    num_epochs = config['optimizer']['schedule']['epochs']

    optimizer1 = torch.optim.Adam(params=[model.MAT.parameters(), model.static.parameters()], lr=lr, weight_decay=0)
    train_loop(model = model, num_epoch=num_epoch, sampler=sampler_train,
            loader_train=loader_train, loader_val=loader_val,
            optimizer=optimizer1, loss_func=twoPhaseLoss(phase=1).to(device),
            device=device,
            phase=1)
    optimizer2 = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=0)
    train_loop(model = model, num_epoch=num_epoch - num_epoch / 2, sampler=sampler_train,
            loader_train=loader_train, loader_val=loader_val,
            optimizer=optimizer2, loss_func=twoPhaseLoss(phase=2).to(device),
            device=device,
            phase=2)

if __name__ == '__main__':
    main()

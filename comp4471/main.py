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
def train_loop(start_epoch, num_epoch, sampler, loader_train, loader_val):
    for epoch in range(num_epochs):
        #if is_distributed:
        sampler.set_epoch(epoch)
        sampler.dataset.next_epoch()
        train_epoch(loader_train)
        validate_epoch(loader_val)

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
                              sampler=train_sampler,
                              pin_memory=False,
                              num_workers=args.workers,
                              )

    loader_val = DataLoader(data_val,
                            batch_size=config.batch_size * 2, # ???
                            sampler=sampler_train,
                            suffle=False,
                            pin_memory=True,
                            num_workers=args.workers,
                            )
def main():
    args = my_parse_args()
    config = load_config(args.config)

    # Load classifier into model
    # model = classifiers

    # Configure loss
    # loss =

    # Configure data
    configure_data(args, config)

    # Train loop
    start_epoch = 0
    # Resume
    num_epochs = conf['optimizer']['schedule']['epochs']
    train_loop(start_epoch, num_epoch)

if __name__ == '__main__':
    main()

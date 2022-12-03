import argparse
import os
import random

import numpy as np
import cv2
from tqdm import tqdm
import pynvml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.distributed.optim import ZeroRedundancyOptimizer as ZeRO
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.tensorboard import SummaryWriter

from comp4471 import loss
from comp4471.train import train_loop
from config import load_config
from dataloader.loader import configure_data
import comp4471.model
import comp4471.ckpt

def my_parse_args():
    parser = argparse.ArgumentParser("ASRID")
    parser.add_argument('--comment', type=str)
    parser.add_argument('--config', metavar='CONFIG_FILE', help='path to configuration file', default='conf.json')

    # parser.add_argument('--board_path', type=str, help='path to tensorboard folder', default='runs/')
    parser.add_argument('--ckpt_path', type=str, help='path to checkpoint file', default=None)
    parser.add_argument('--workers', type=int, default=1, help='number of cpu threads to use for data loading')

    parser.add_argument('--is-distributed', help='is distributed', action='store_true') # action-> having means True
    # parser.add_argument('--backend', type=str, default='nccl', choices=['gloo', 'nccl'])
    parser.add_argument('--gpu-workers', type=int, default=4, help='number of GPU')

    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--root-dir', type=str, default="data/")
    parser.add_argument('--folds-csv-path', type=str, default='folds.csv')
    parser.add_argument('--folds-json-path', type=str, default='folds.json')
    parser.add_argument('--crops-dir', type=str, default='crops')
    parser.add_argument('--output-dir', type=str, default='weights/')
    return parser.parse_args()

def setup_seed(seed = 4471):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main_worker(worker_id, world_size, gpus, args, config):
    if worker_id is None:
        device = torch.device("cpu")
    else:
        device = gpus[worker_id]
        print(f'worker {worker_id}/{world_size} are using GPU {device}/{len(gpus)}')
        if args.is_distributed:
            dist.init_process_group(backend='nccl', init_method='tcp://localhost:2024', world_size=world_size, rank=worker_id)

        torch.cuda.device(device)
        torch.cuda.empty_cache()
        torch.backends.cudnn.benchmark = True # https://zhuanlan.zhihu.com/p/73711222

    if args.is_distributed:
        if worker_id == 0:
            comp4471.model.get_pretrained()
            print('multi-process unsafe operations finished')
        print(f'worker {worker_id} before barrier')
        torch.distributed.barrier()
        print(f'worker {worker_id} after barrier')
    else:
        comp4471.model.get_pretrained()

    # Resume
    lr = config['optimizer']['lr']
    weight_delay = config['optimizer']['weight_decay']
    batch_size = config['optimizer']['batch_size']
    num_epoch = config['optimizer']['schedule']['num_epoch']
    schedule_policy = config['optimizer']['schedule']['schedule_policy']

    # Configure data
    root_path = os.path.dirname(__file__)
    _, loader_train, loader_val = configure_data(args, config)
    print('data configuration finish')

    # Load model
    model = comp4471.model.ASRID(batch_size=batch_size).to(device)
    if args.is_distributed:
        optimizer = ZeRO(params=[
            {'params': model.efficientNet.parameters()},
            {'params': model.multiattn_block.parameters()},
            {'params': model.static_block.parameters()}
        ], lr=lr, weight_decay=weight_delay, optimizer_class=torch.optim.AdamW)
        # https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer
    else:
        optimizer = torch.optim.AdamW(params=[
            {'params': model.efficientNet.parameters()},
            {'params': model.multiattn_block.parameters()},
            {'params': model.static_block.parameters()}
        ], lr=lr, weight_decay=weight_delay)

    if args.is_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)
    print('model definition finish')

    # State
    if args.ckpt_path is None:
        args.ckpt_path = os.path.join(root_path, 'ckpt', 'default.pth.tar')
    state = comp4471.ckpt.State(model=model, optimizer=optimizer, ckpt_path=args.ckpt_path)
    torch_device = torch.device("cpu") if worker_id is None else torch.device('cuda', device)
    state.load(torch_device)
    print('state restore finish')

    # Writer
    # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_custom_scalars
    # - purge_step: redo experiment from step [purge_step]
    # - comment: show the stage/usage of experiments, e.g. LR_0.1_BATCH_16
    # - log_dir: use default(runs/CURRENT_DATETIME_HOSTNAME)
    if (args.is_distributed and worker_id == 0) or not args.is_distributed:
        writer = SummaryWriter(comment = f'LR{lr}_Wdecay{weight_delay}_{schedule_policy}_{args.comment}', purge_step=state.iter)
        layout = {
            "MyLayout": {
                "His": ["Multiline", ["His/loss", "His/val"]],
            },
        }
        writer.add_custom_scalars(layout)
    else:
        writer = None
    print('writer ready')

    train_loop(state, central_gpu=gpus[0] if worker_id!=None else device,
        model=model, num_epoch=num_epoch, device=device, writer=writer,
        loader_train=loader_train, loader_val=loader_val,
        optimizer=optimizer, schedule_policy=schedule_policy,
        loss_func=loss.twoPhaseLoss(phase=1).to(device), eval_func=loss.evalLoss().to(device),
        distributed=args.is_distributed, phase=1)

    #optimizer2 = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_delay)
    #start_epoch, start_iter = train_loop(state,
    #    model=model, num_epoch=num_epoch - num_epoch / 2, device=device, writer=writer,
    #    sampler_train=sampler_train, loader_train=loader_train, loader_val=loader_val,
    #    optimizer=optimizer2, schedule_policy=schedule_policy,
    #    loss_func=loss.twoPhaseLoss(phase=2).to(device), eval_func=loss.evalLoss().to(device),
    #    start_epoch=start_epoch, phase=2,
    #    start_iter=start_iter) # kwargs

    # clean up
    torch.cuda.empty_cache()
    dist.destroy_process_group()

@record
def main():
    args = my_parse_args()
    config = load_config(args.config)
    setup_seed(config['seed'])

    if torch.cuda.is_available() is False:
        print('No GPU found, use CPU instead')
        main_worker(None, None, None, args, config)
        return

    # sort most available GPUs
    ngpus_per_node = torch.cuda.device_count()
    gpus = range(0, ngpus_per_node)

    if args.is_distributed:
        assert dist.is_available() and dist.is_nccl_available()
        try:
            local_rank = int(os.environ["LOCAL_RANK"])
            ngpus_per_node = int(os.environ["WORLD_SIZE"])
            torch_run = True
        except:
            torch_run = False

        if torch_run:
            # option 1: use torch run (better)
            # Worker failures are handled gracefully by restarting all workers.
            main_worker(local_rank, ngpus_per_node, gpus, args, config)
        else:
            # option 2: use mp to use less GPU when others are busy
            # If one of the processes exits with a non-zero exit status, the remaining processes are killed and an exception is raised with the cause of termination.
            mp.spawn(main_worker, nprocs=args.gpu_workers, args=(args.gpu_workers, gpus, args, config))
    else:
        main_worker(0, 1, gpus, args, config)

if __name__ == '__main__':
    main()

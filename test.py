import torch
from omegaconf import OmegaConf

from comp4471 import loss
from comp4471.train import train_loop
from dataloader.loader import configure_data
import comp4471.model
import comp4471.ckpt
import comp4471.util

def main(conf: OmegaConf):
    # https://omegaconf.readthedocs.io/en/2.2_branch/usage.html
    try:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg.conf_file)) # merge with config file content
    except:
        pass

    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    cfg_seed = OmegaConf.masked_copy(cfg, ["strategy", "optimizer", "schedule","dataset"])
    seed = comp4471.util.hash8(OmegaConf.to_yaml(cfg_seed))
    comp4471.util.setup_seed(seed)
    print(f'using seed {seed} by hashing config yaml')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Resume
    lr = cfg['optimizer']['lr']
    weight_delay = cfg['optimizer']['weight_decay']
    batch_size = cfg['optimizer']['batch_size']

    # Configure data
    root_path = os.path.dirname(__file__)
    loader_train, loader_test = configure_data(cfg)
    loader_test = configure_data_test(cfg)
    print('data configuration finish')

    # Load model
    model = comp4471.model.ASRID(batch_size=batch_size, strategy=cfg["strategy"]).to(device)
    optimizer = torch.optim.AdamW(params=[
        {'params': model.efficientNet.parameters()},
        {'params': model.multiattn_block.parameters()},
        {'params': model.static_block.parameters()}
    ], lr=lr, weight_decay=weight_delay)

    state = comp4471.ckpt.State(model=model, optimizer=optimizer, ckpt_path=cfg['experiment']['ckpt_path'])
    state.load(device)
    loss_func=loss.twoPhaseLoss(phase=1).to(device)
    eval_func=loss.evalLoss().to(device)
    metric = validate_epoch(model=model, device=device, data_loader=loader_test, verbose=True, eval_func=eval_func)
    print(metric)

# python test.py conf_file=./exps/exp1.yaml
if __name__ == '__main__':
    main(OmegaConf.from_cli())



import argparse
import os


import numpy as np
import cv2
from tqdm import tqdm
import pynvml
import shutil
from omegaconf import OmegaConf

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
from dataloader.loader import configure_data
import comp4471.model
import comp4471.ckpt
import comp4471.util

def main_worker(worker_id, world_size, gpus, cfg):
    is_distributed = cfg['distributed']['toggle']
    print(f'worker {worker_id} waking up with distributed {is_distributed}')
    if worker_id is None:
        device = torch.device("cpu")
    else:
        device = gpus[worker_id]
        print(f'worker {worker_id}/{world_size} are using GPU {device}/{len(gpus)} (relative)')
        if is_distributed:
            print('initializing distributed group')
            dist.init_process_group(backend='nccl', init_method=cfg['distributed']['server']['url'], world_size=world_size, rank=worker_id)
            print('initialized distributed group')
        torch.cuda.device(device)

    if is_distributed:
        if worker_id == 0:
            comp4471.model.get_pretrained()
            print('multi-process unsafe operations finished')
        print(f'worker {worker_id} before barrier')
        torch.distributed.barrier()
        print(f'worker {worker_id} after barrier')
    else:
        comp4471.model.get_pretrained()

    # Resume
    lr = cfg['optimizer']['lr']
    weight_delay = cfg['optimizer']['weight_decay']
    batch_size = cfg['optimizer']['batch_size']

    # Configure data
    root_path = os.path.dirname(__file__)
    loader_train, loader_val = configure_data(cfg)
    print('data configuration finish')

    # Load model
    model = comp4471.model.ASRID(batch_size=batch_size, strategy=cfg["strategy"]).to(device)
    if is_distributed:
        optimizer = ZeRO(params=[
            {'params': model.parameters()},
        ], lr=lr, weight_decay=weight_delay, optimizer_class=torch.optim.AdamW)
        # https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer
    else:
        optimizer = torch.optim.AdamW(params=[
            {'params': model.parameters()},
        ], lr=lr, weight_decay=weight_delay)

    if is_distributed:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device, find_unused_parameters=True)
    print('model definition finish')

    # State
    state = comp4471.ckpt.State(model=model, optimizer=optimizer, ckpt_path=cfg['experiment']['ckpt_path'])
    torch_device = torch.device("cpu") if worker_id is None else torch.device('cuda', device)
    state.load(torch_device)
    print('state restore finish')




    # clean up
    # torch.cuda.empty_cache()
    # https://discuss.pytorch.org/t/out-of-memory-when-i-use-torch-cuda-empty-cache/57898
    dist.destroy_process_group()

@record
def main(cfg: OmegaConf):
    # https://omegaconf.readthedocs.io/en/2.2_branch/usage.html
    try:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(cfg.conf_file)) # merge with config file content
    except:
        pass

    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    cfg_seed = OmegaConf.masked_copy(cfg, ["strategy", "optimizer", "schedule","dataset"])
    seed = comp4471.util.hash8(OmegaConf.to_yaml(cfg_seed))
    comp4471.util.setup_seed(seed)
    print(f'using seed {seed} by hashing config yaml')

    if torch.cuda.is_available() is False:
        print('No GPU found, use CPU instead')
        main_worker(None, None, None, config)
        return

    # find GPUs
    ngpus_per_node = torch.cuda.device_count()
    gpus = range(0, ngpus_per_node)
    print(f'{ngpus_per_node} GPUs found')

    if cfg['distributed']['toggle']:
        assert dist.is_available() and dist.is_nccl_available()
        try:
            local_rank = int(os.environ["LOCAL_RANK"])
            ngpus_per_node = int(os.environ["WORLD_SIZE"])
            torch_run = True
        except:
            torch_run = False
        print(f'running in torch run = {torch_run}')

        if torch_run:
            # option 1: use torch run (better)
            # Worker failures are handled gracefully by restarting all workers.
            main_worker(local_rank, ngpus_per_node, gpus, cfg)
        else:
            # option 2: use mp to use less GPU when others are busy
            # If one of the processes exits with a non-zero exit status, the remaining processes are killed and an exception is raised with the cause of termination.
            mp.spawn(main_worker, nprocs=cfg['distributed']['gpu_workers'], args=(cfg['distributed']['gpu_workers'], gpus, cfg))
    else:
        main_worker(0, 1, gpus, cfg)

# python main.py conf_file=./exps/exp1.yaml
if __name__ == '__main__':
    main(OmegaConf.from_cli())

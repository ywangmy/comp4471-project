import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.tensorboard
from torch.distributed.optim import ZeroRedundancyOptimizer as ZeRO

import torchvision
import torchvision.datasets

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from comp4471 import loss
from comp4471.model import ASRID
from comp4471.train import train_loop
from config import load_config
from dataloader.loader import configure_data

def evaluate(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return accuracy

import shutil
def save_checkpoint(state: State, is_best: bool, filename: str):
    checkpoint_dir = os.path.dirname(filename)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # save to tmp, then commit by moving the file in case the job gets interrupted while writing the checkpoint
    tmp_filename = filename + ".tmp"
    torch.save(state.capture_snapshot(), tmp_filename)
    os.rename(tmp_filename, filename)
    print(f"=> saved checkpoint for epoch {state.epoch} at {filename}")
    if is_best:
        best = os.path.join(checkpoint_dir, "model_best.pth.tar")
        print(f"=> best model found at epoch {state.epoch} saving to {best}")
        shutil.copyfile(filename, best)

# device: id in [0, world_size)
def main_worker(id, world_size, gpus):
    local_rank = gpus[id][1]
    print(f'worker {id}/{world_size} are using GPU {local_rank}')
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:2333', world_size=world_size, rank=local_rank)
    torch.cuda.set_device(local_rank)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 10),
        nn.Softmax()
    ).to(local_rank)
    model = nn.parallel.DistributedDataParallel(model)

    resume = False
    if resume is True:
        map_location = {"cuda:0": "cuda:{}".format(local_rank)}
        model.load_state_dict(torch.load(model_filepath, map_location=map_location))

    root_path = os.path.dirname(__file__)
    dataset_path = os.path.join(root_path, 'dataset', 'FashionMNIST')

    print('checkpoint 0')
    if id == 0: # multi-process unsafe operations
        torchvision.datasets.FashionMNIST(dataset_path, train=True, download=True)
        torchvision.datasets.FashionMNIST(dataset_path, train=False, download=True)
        print('id 0 finished')
    torch.distributed.barrier()

    print('checkpoint 1')

    batch_size = 1024

    data_train = torchvision.datasets.FashionMNIST(dataset_path, train=True, download=False, transform = torchvision.transforms.ToTensor())
    data_val = torchvision.datasets.FashionMNIST(dataset_path, train=False, download=False, transform = torchvision.transforms.ToTensor())
    sampler_train = torch.utils.data.distributed.DistributedSampler(dataset=data_train)
    loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch_size, sampler=sampler_train, num_workers=world_size)
    loader_val = torch.utils.data.DataLoader(dataset=data_val, batch_size=batch_size, shuffle=False, num_workers=world_size)

    print('checkpoint 2')

    num_epochs = 4
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:{}".format(local_rank))

    writer = torch.utils.tensorboard.SummaryWriter()
    # optimizer = torch.optim.AdamW(params=model.parameters())
    # https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer
    optimizer = ZeRO(params=model.parameters(), optimizer_class=torch.optim.AdamW)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, epochs=num_epochs, steps_per_epoch=len(loader_train), last_epoch=-1, cycle_momentum=False) # AdamW no momentum

    #start_epoch, start_iter = train_loop(
    #    model = model, num_epoch=1, device=local_rank, writer=writer,
    #    sampler_train=sampler_train, loader_train=loader_train, loader_val=loader_val,
    #    optimizer=optimizer2, schedule_policy=schedule_policy,
    #    loss_func=loss.twoPhaseLoss(phase=2).to(device), eval_func=loss.evalLoss().to(device),
    #    start_epoch=start_epoch, phase=2,
    #    start_iter=start_iter) # kwargs

    for epoch in range(num_epochs):

        print("Local Rank: {}, Epoch: {}, Training ...".format(local_rank, epoch))

        # Save and evaluate model routinely
        if epoch % 1 == 0:
            if local_rank == 0:
                accuracy = evaluate(model=model, device=device, test_loader=loader_val)
                #torch.save(model.state_dict(), model_filepath)
                print("-" * 75)
                print("Epoch: {}, Accuracy: {}".format(epoch, accuracy))
                print("-" * 75)

        model.train()
        for data in loader_train:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            lr_scheduler.step()

        if id == 0:
            is_best = acc1 > state.best_acc1
            state.best_acc1 = max(acc1, state.best_acc1)
            save_checkpoint(state, is_best, args.checkpoint_file)

    #ddp_logging_data = model._get_ddp_logging_data()
    #assert ddp_logging_data.get("can_set_static_graph") == True
    dist.destroy_process_group()

import pynvml
if __name__ == '__main__':
    assert dist.is_available() and dist.is_nccl_available()
    ngpus_per_node = torch.cuda.device_count()

    use_torchrun = False
    if use_torchrun: # option 1: use torch run
        local_rank = int(os.environ["LOCAL_RANK"])
        main_worker(local_rank, ngpus_per_node)
    else: # option 2: use mp to use less GPU when others are busy
        torchvision.datasets.FashionMNIST('../dataset/FashionMNIST', download=True)
        # If one of the processes exits with a non-zero exit status, the remaining processes are killed and an exception is raised with the cause of termination.

        gpus = []
        pynvml.nvmlInit()
        for i in range(0, ngpus_per_node):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            print(f'GPU {i}:{info.free / 1024 ** 2} free, {info.used / 1024 ** 2} used')
            gpus.append((info.free, i))
        gpus.sort(reverse=True)
        while len(gpus)>0:
            last = gpus[-1]
            if last[0] < 2 * 1024**3: gpus.pop()
            else: break
        ngpus_per_node = len(gpus)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,gpus))
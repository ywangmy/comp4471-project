import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.tensorboard
from torch.distributed.optim import ZeroRedundancyOptimizer as ZeRO
from torch.distributed.elastic.multiprocessing.errors import record

import torchvision
import torchvision.datasets

import sys, os, time
from datetime import timedelta
from typing import List, Tuple

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from comp4471 import loss
from comp4471.model import ASRID
from comp4471.train import train_loop
from config import load_config
from dataloader.loader import configure_data



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

class State:
    """
    Container for objects that we want to checkpoint. Represents the
    current "state" of the worker. This object is mutable.
    """

    def __init__(self, model, optimizer):
        self.epoch = -1
        self.best_acc1 = 0
        self.model = model
        self.optimizer = optimizer

    def capture_snapshot(self):
        """
        Essentially a ``serialize()`` function, returns the state as an
        object compatible with ``torch.save()``. The following should work
        ::
        snapshot = state_0.capture_snapshot()
        state_1.apply_snapshot(snapshot)
        assert state_0 == state_1
        """
        return {
            "epoch": self.epoch,
            "best_acc1": self.best_acc1,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

    def apply_snapshot(self, obj, device_id):
        """
        The complimentary function of ``capture_snapshot()``. Applies the
        snapshot object that was returned by ``capture_snapshot()``.
        This function mutates this state object.
        """

        self.epoch = obj["epoch"]
        self.best_acc1 = obj["best_acc1"]
        # self.state_dict = obj["state_dict"]
        self.model.load_state_dict(obj["state_dict"])
        self.optimizer.load_state_dict(obj["optimizer"])

    def save(self, f):
        torch.save(self.capture_snapshot(), f)

    def load(self, f, device_id):
        # Map model to be loaded to specified single gpu.
        snapshot = torch.load(f, map_location=f"cuda:{device_id}")
        self.apply_snapshot(snapshot, device_id)

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

def load_checkpoint(
    checkpoint_file: str,
    device_id: int,
    model: nn.parallel.DistributedDataParallel,
    optimizer,  # SGD
) -> State:
    state = State(model, optimizer)

    if os.path.isfile(checkpoint_file):
        print(f"=> loading checkpoint file: {checkpoint_file}")
        state.load(checkpoint_file, device_id)
        print(f"=> loaded checkpoint file: {checkpoint_file}")
    print(f"=> done restoring from previous checkpoint")
    return state


# device: id in [0, world_size)
@record
def main_worker(id, world_size, gpus):
    local_rank = gpus[id]
    print(f'worker {id}/{world_size} are using GPU {local_rank}')
    dist.init_process_group(backend='nccl', init_method='tcp://localhost:2333', world_size=world_size, rank=local_rank)
    torch.cuda.set_device(local_rank)
    device = local_rank

    # https://zhuanlan.zhihu.com/p/73711222
    torch.backends.cudnn.benchmark = True

    # model
    num_epochs = 10
    criterion = nn.CrossEntropyLoss().to(device)
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 10),
        nn.Softmax()
    ).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[device], output_device=device)

    # optimizer = torch.optim.AdamW(params=model.parameters())
    # https://pytorch.org/docs/stable/distributed.optim.html#torch.distributed.optim.ZeroRedundancyOptimizer
    optimizer = ZeRO(params=model.parameters(), optimizer_class=torch.optim.AdamW)

    # data
    root_path = os.path.dirname(__file__)
    dataset_path = os.path.join(root_path, 'dataset', 'FashionMNIST')
    ckpt_path = os.path.join(root_path, 'ckpt', 'checkpoint.pth.tar')
    batch_size = 1024

    if id == 0: # multi-process unsafe operations
        torchvision.datasets.FashionMNIST(dataset_path, train=True, download=True)
        torchvision.datasets.FashionMNIST(dataset_path, train=False, download=True)
        print('id 0 finished')
    torch.distributed.barrier()

    data_train = torchvision.datasets.FashionMNIST(dataset_path, train=True, download=False, transform = torchvision.transforms.ToTensor())
    data_val = torchvision.datasets.FashionMNIST(dataset_path, train=False, download=False, transform = torchvision.transforms.ToTensor())
    sampler_train = torch.utils.data.distributed.DistributedSampler(dataset=data_train)
    loader_train = torch.utils.data.DataLoader(dataset=data_train, batch_size=batch_size, sampler=sampler_train, pin_memory=True,num_workers=world_size)
    loader_val = torch.utils.data.DataLoader(dataset=data_val, batch_size=batch_size, shuffle=False, pin_memory=True,num_workers=world_size)

    # state
    state = load_checkpoint(
        ckpt_path, device, model, optimizer
    )
    start_epoch = state.epoch + 1
    print(f"=> start_epoch: {start_epoch}, best_acc1: {state.best_acc1}")


    writer = torch.utils.tensorboard.SummaryWriter()

    #start_epoch, start_iter = train_loop(
    #    model = model, num_epoch=1, device=device, writer=writer,
    #    sampler_train=sampler_train, loader_train=loader_train, loader_val=loader_val,
    #    optimizer=optimizer2, schedule_policy=schedule_policy,
    #    loss_func=loss.twoPhaseLoss(phase=2).to(device), eval_func=loss.evalLoss().to(device),
    #    start_epoch=start_epoch, phase=2,
    #    start_iter=start_iter) # kwargs



    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-2, epochs=num_epochs, steps_per_epoch=len(loader_train), last_epoch=-1, cycle_momentum=False) # AdamW no momentum
    for epoch in range(start_epoch, start_epoch+num_epochs):
        print("Local Rank: {}, Epoch: {}, Training ...".format(device, epoch))

        state.epoch = epoch
        loader_train.batch_sampler.sampler.set_epoch(epoch)

        batch_time = AverageMeter("Time", ":6.3f")
        data_time = AverageMeter("Data", ":6.3f")
        losses = AverageMeter("Loss", ":.4e")
        progress = ProgressMeter(
            len(loader_train),
            [batch_time, data_time, losses],
            prefix="Epoch: [{}]".format(epoch),
        )
        model.train()
        end = time.time()
        for i, data in enumerate(loader_train):
            data_time.update(time.time() - end)
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            losses.update(loss.item(), labels.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_scheduler.step()

            batch_time.update(time.time() - end)
            end = time.time()
            if i % 100 == 0:
                progress.display(i)

        optimizer.consolidate_state_dict(to=gpus[0]) # send state_dict to device 0
        # Save and evaluate model routinely
        if epoch % 1 == 0:
            if id == 0:
                acc1 = evaluate(model=model, device=device, test_loader=loader_val)
                print("-" * 75)
                print("Epoch: {}, Accuracy: {}".format(epoch, acc1))
                print("-" * 75)

                is_best = acc1 > state.best_acc1
                state.best_acc1 = max(acc1, state.best_acc1)
                save_checkpoint(state, is_best, ckpt_path)

    #ddp_logging_data = model._get_ddp_logging_data()
    #assert ddp_logging_data.get("can_set_static_graph") == True
    dist.destroy_process_group()

import pynvml
if __name__ == '__main__':
    assert dist.is_available() and dist.is_nccl_available()
    ngpus_per_node = torch.cuda.device_count()

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        ngpus_per_node = int(os.environ["WORLD_SIZE"])
        torch_run = True
    except:
        torch_run = False

    if torch_run:
        # option 1: use torch run
        main_worker(local_rank, ngpus_per_node, range(0,ngpus_per_node))
    else:
        # option 2: use mp to use less GPU when others are busy
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
        gpus = [i for capability, i in gpus ]
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node,gpus))


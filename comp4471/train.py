import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR

#def save():
#    torch.save(model.state_dict(), "model.pth")
#    print("Saved PyTorch Model State to model.pth")

#def load():
#    model = ASRID()
#    model.load_state_dict(torch.load("model.pth"))

# train for single epoch
def train_epoch(model, device, writer, data_loader,
                optimizer, loss_func, start_iter, phase, **kwargs):
    # Keyword arguments:
    show_every = kwargs.get('show_every', 100)
    total_iter = len(data_loader)

    model.train()
    #for iter, (X, y) in enumerate(data_loader):
    for iter, sample in enumerate(data_loader):
        # print(f'iter = {iter}, total {torch.cuda.get_device_properties(0).total_memory / 1024**2}, alloc {torch.cuda.memory_allocated(device) / 1024**2}, maxalloc {torch.cuda.max_memory_allocated(device) / 1024**2}, reserved {torch.cuda.memory_reserved(0) / 1024**2}')

        # X is images
        X = sample["image"].float()
        ## y is labels
        y = sample["labels"].float()
        X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True) # y should be float()
        score, attn_output = model(X)

        ##writer.add_images('X', X.transpose(1,2), global_step=start_iter+iter, dataformats='CHW')
        ##writer.flush()

        score = score.view(-1, 1)
        loss = loss_func(score, y).mean()
        writer.add_scalar(f'Loss/train{phase}', loss, start_iter+iter)
        for i, param_group in enumerate(optimizer.param_groups):
            writer.add_scalar(f'Lr/lr{phase}-param{i}', param_group['lr'], start_iter+iter)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter == 0 or iter % show_every == 0:
            print(f'iter {start_iter+iter}/{start_iter+total_iter}: loss {loss}')
    return start_iter + total_iter

def validate_epoch(model, device, writer, data_loader,
                eval_func, **kwargs):
    model.eval()
    with torch.no_grad():
        sum, capacity = 0., 0
        for iter, sample in enumerate(data_loader):
            X = sample["image"].float()
            y = sample["labels"].float()
            X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            score, attn_output = model(X)
            score = score.view(-1, 1)
            sum += eval_func(score, y)
            capacity += X.shape[0]
        return sum / capacity

def train_loop(model, num_epoch, device, writer,
            sampler_train, loader_train, loader_val,
            optimizer, lr_scheduler,
	    loss_func, eval_func,
            start_epoch = 0, start_iter = 0, val_freq = 1, verbose = True, phase = 1, **kwargs):
    for epoch in range(start_epoch, start_epoch+num_epoch):
        if verbose: print(f'epoch {epoch} in progress')
        if sampler_train is not None:
            sampler_train.set_epoch(epoch)
            sampler_train.dataset.next_epoch()

        start_iter = train_epoch(model, device, writer, loader_train, optimizer, loss_func, start_iter, phase, kwargs=kwargs)
        if epoch % val_freq == 0:
            metric = validate_epoch(model, device, writer, loader_val, eval_func, kwargs=kwargs)
            if verbose: print(f'eval{phase} = {metric}')
            writer.add_scalar(f'Eval/val{phase}', metric, epoch)
            lr_scheduler.step(metrics=metric, epoch=epoch)
    return start_epoch+num_epoch, start_iter

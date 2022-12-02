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
def train_epoch(model, device, data_loader, writer,
                optimizer, phase, loss_func,
                epoch, start_iter, schedule_policy, lr_scheduler,
                **kwargs):
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
        if schedule_policy == 'Cosine': lr_scheduler.step(epoch + iter / total_iter)
        elif schedule_policy == 'Plateau': pass
        elif schedule_policy == 'OneCycle': lr_scheduler.step()

        if iter == 0 or iter % show_every == 0:
            print(f'iter {start_iter+iter}/{start_iter+total_iter}: loss {loss}')
    return start_iter + total_iter

def validate_epoch(model, device, data_loader,
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
            optimizer, schedule_policy,
            loss_func, eval_func,
            start_epoch = 0, start_iter = 0, val_freq = 1, verbose = True, phase = 1, **kwargs):
    if schedule_policy == 'Cosine':
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=0, last_epoch=start_epoch-1)
    elif schedule_policy == 'Plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    elif schedule_policy == 'OneCycle':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-5, epochs=num_epoch, steps_per_epoch=len(loader_train), last_epoch=start_epoch-1)

    for epoch in range(start_epoch, start_epoch+num_epoch):
        if verbose: print(f'epoch {epoch} in progress')
        if sampler_train is not None:
            sampler_train.set_epoch(epoch)
            sampler_train.dataset.next_epoch()

        start_iter = train_epoch(model=model, device=device, data_loader=loader_train, writer=writer,
                optimizer=optimizer, phase=phase, loss_func=loss_func,
                epoch=epoch, start_iter=start_iter, schedule_policy=schedule_policy, lr_scheduler=lr_scheduler, kwargs=kwargs)
        if epoch % val_freq == 0:
            metric = validate_epoch(model=model, device=device, data_loader=loader_val, eval_func=eval_func, kwargs=kwargs)
            if verbose: print(f'eval{phase} = {metric}')
            writer.add_scalar(f'Eval/val{phase}', metric, epoch)
            if schedule_policy == 'Cosine': pass
            elif schedule_policy == 'Plateau': lr_scheduler.step(metrics=metric)
            elif schedule_policy == 'OneCycle': pass

    return start_epoch+num_epoch, start_iter

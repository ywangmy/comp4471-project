import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from datetime import timedelta
import time

from comp4471.eval import AverageMeter, ProgressMeter

# train for single epoch
def train_epoch(model, device, data_loader,
                verbose, writer,
                optimizer, phase, loss_func,
                epoch, start_iter, schedule_policy, lr_scheduler,
                **kwargs):
    # Keyword arguments:
    show_every = kwargs.get('show_every', 20)
    total_iter = len(data_loader)

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    progress = ProgressMeter(
        start_iter + total_iter,
        [batch_time, data_time],
    )

    model.train()
    end = time.time()
    for iter, sample in enumerate(data_loader):
        data_time.update(time.time() - end)
        # print(f'iter = {iter}, total {torch.cuda.get_device_properties(0).total_memory / 1024**2}, alloc {torch.cuda.memory_allocated(device) / 1024**2}, maxalloc {torch.cuda.max_memory_allocated(device) / 1024**2}, reserved {torch.cuda.memory_reserved(0) / 1024**2}')

        X = sample["video"].float().to(device, non_blocking=True)
        y = sample["label"].float().to(device, non_blocking=True) # y should be float()
        # video_name = sample['video_name']
        # ori_name = sample['ori']
        # writer.add_images('X', X.transpose(1,2), global_step=start_iter+iter, dataformats='CHW')
        # writer.flush()

        score, attn_output = model(X)
        loss = loss_func(score.view(-1, 1), y)

        if writer is not None:
            writer.add_scalar(f'His/loss', loss, start_iter+iter)
            for i, param_group in enumerate(optimizer.param_groups):
                writer.add_scalar(f'His/lr', param_group['lr'], start_iter+iter)
                break

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if schedule_policy == 'Cosine': lr_scheduler.step(epoch + iter / total_iter)
        elif schedule_policy == 'Plateau': pass
        elif schedule_policy == 'OneCycle': lr_scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if (iter == 0 or iter % show_every == 0) and verbose:
            print(progress.display(start_iter+iter))
    return start_iter + total_iter

def validate_epoch(model, device, data_loader, verbose,
                eval_func, **kwargs):
    model.eval()
    losses = AverageMeter("Loss", ":.4e")
    val_time = AverageMeter("Time", ":6.3f")
    progress = ProgressMeter(
        len(data_loader),
        [losses, val_time],
    )
    with torch.no_grad():
        for iter, sample in enumerate(data_loader):
            X = sample["video"].float().to(device, non_blocking=True)
            y = sample["label"].float().to(device, non_blocking=True)
            #video_name = sample['video_name']
            #ori_name = sample['ori']
            end = time.time()

            score, attn_output = model(X)
            loss = eval_func(score.view(-1, 1), y)

            val_time.update(time.time() - end)
            losses.update(loss.item(), X.size(0))
            if iter + 1 == len(data_loader): print(progress.display(0))
        return losses.avg

def train_loop(state, central_gpu,
            model, num_epoch, device, writer,
            loader_train, loader_val,
            optimizer, schedule_policy,
            loss_func, eval_func,
            distributed,
            val_freq = 1, verbose = True, phase = 1, **kwargs):
    start_iter = state.iter
    start_epoch = state.epoch
    if writer is None: verbose = False

    # learning rate scheduler
    if schedule_policy == 'Cosine':
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=0, last_epoch=start_epoch-1)
    elif schedule_policy == 'Plateau':
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=2, threshold=0.01, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
    elif schedule_policy == 'OneCycle':
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-6, epochs=num_epoch, steps_per_epoch=len(loader_train), last_epoch=start_iter-1, cycle_momentum=False)

    for epoch in range(start_epoch, start_epoch + num_epoch):
        if verbose: print(f'epoch {epoch} in progress')

        loader_train.batch_sampler.sampler.set_epoch(epoch)
        #sampler_train.set_epoch(epoch)
        #sampler_train.dataset.next_epoch()

        start_iter = train_epoch(model=model, device=device, data_loader=loader_train,
                verbose=verbose, writer=writer,
                optimizer=optimizer, phase=phase, loss_func=loss_func,
                epoch=epoch, start_iter=start_iter, schedule_policy=schedule_policy, lr_scheduler=lr_scheduler, kwargs=kwargs)

        if distributed:
            optimizer.consolidate_state_dict(to=central_gpu) # send state_dict to worker 0
        if epoch % val_freq == 0 and writer is not None: # holder of writer means worker 0
            metric = validate_epoch(model=model, device=device, data_loader=loader_val, verbose=verbose, eval_func=eval_func, kwargs=kwargs)

            if verbose: print(f'eval{phase} = {metric}')
            writer.add_scalar(f'His/val', metric, start_iter-1)
            if schedule_policy == 'Cosine': pass
            elif schedule_policy == 'Plateau': lr_scheduler.step(metrics=metric)
            elif schedule_policy == 'OneCycle': pass

            state.epoch = epoch + 1
            state.iter = start_iter + 1
            state.save(metric=metric)

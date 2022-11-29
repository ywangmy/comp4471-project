import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


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
    show_every = kwargs.pop('show_every', 100)
    start_iter = kwargs.pop('start_iter', 0)
    total_iter = len(data_loader)

    model.train()
    for iter, (X, y) in enumerate(data_loader):
        X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        score = model(X)
        loss = loss_func(score, y).mean()
        writer.add_scalar(f'Loss/train{phase}', loss, start_iter+iter)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter == 0 or iter % show_every == 0:
            print(f'iter {start_iter+iter}/{start_iter+total_iter}: loss {loss}')

def validate_epoch(model, device, writer, data_loader,
                eval_func, **kwargs):
    model.eval()
    with torch.no_grad():
        sum, capacity = 0., 0
        for iter, (X, y) in enumerate(data_loader):
            X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            sum += eval_func(model(X), y).mean()
            capacity += X.shape[0]
        return sum / capacity

def train_loop(model, num_epoch, device, writer,
            sampler_train, loader_train, loader_val,
            optimizer, loss_func, eval_func,
            start_epoch = 0, val_freq = 2, verbose = True, phase = 1, **kwargs):
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, verbose=True)
    for epoch in range(num_epochs):
        if verbose:
            print(f'epoch {start_epoch + epoch} in progress')
        #if is_distributed:
        sampler_train.set_epoch(epoch)
        sampler_train.dataset.next_epoch()
        train_epoch(model, device, writer, loader_train, optimizer, loss_func, phase, kwargs)
        lr_scheduler.step()
        if epoch % val_freq == 0:
            metric = validate_epoch(model, device, loader_val, eval_func, kwargs)
            print(f'eval{phase} = {metric}')
            writer.add_scalar(f'Eval/val{phase}', metric, epoch)
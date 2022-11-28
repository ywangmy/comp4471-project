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
def train_epoch(model, device, data_loader, optimizer, loss_func, **kwargs):
    # Keyword arguments:
    show_every = kwargs.pop('show_every', 100)

    model.train()
    for iter, (X, y) in enumerate(data_loader):
        X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        score = model(X)
        loss = loss_func(score, y).mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter == 0 or iter % show_every == 0:
            print(f'iter {iter}, loss {loss}')

def validate_epoch(model, device, data_loader, eval_func, **kwargs):
    model.eval()
    with torch.no_grad():
        sum, capacity = 0., 0
        for iter, (X, y) in enumerate(data_loader):
            X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            sum += eval_func(model(X), y).mean()
            capacity += X.shape[0]
        return sum / capacity

def train_loop(model, num_epoch, sampler_train, loader_train, loader_val, optimizer, loss_func, eval_func, device, start_epoch = 0, val_freq = 2, verbose = True, **kwargs):
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, verbose=True)
    for epoch in range(num_epochs):
        if verbose:
            print(f'epoch {start_epoch + epoch} in progress')
        sampler_train.set_epoch(epoch)
        sampler_train.dataset.next_epoch()
        train_epoch(model, device, loader_train, optimizer, loss_func, kwargs)
        lr_scheduler.step()
        if epoch % val_freq == 0:
            metric = validate_epoch(model, device, loader_val, eval_func, kwargs)
            print(f'eval = {metric}')
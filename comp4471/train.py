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
def train_epoch(model, data_loader, optimizer, loss_func, device, **kwargs):
    # train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, summary_writer, conf, args.local_rank, args.only_changed_frames)

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

def validate_epoch(model, data_loader, eval_func, device, **kwargs):
    model.eval()
    with torch.no_grad():
        sum, capacity = 0, 0
        for iter, (X, y) in enumerate(data_loader):
            X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            score = model(X)
            performance = eval_func(score, y).mean()
            sum += performance
            capacity += X.shape[0]
        return sum(performances) / capacity

def train_loop(model, num_epoch, sampler, loader_train, loader_val, optimizer, loss_func, eval_func, device, start_epoch = 0, val_freq = 2, verbose = True, **kwargs):
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, verbose=True)
    for epoch in range(num_epochs):
        if verbose:
            print(f'epoch {start_epoch + epoch} in progress')
        sampler.set_epoch(epoch)
        sampler.dataset.next_epoch()
        train_epoch(model, loader_train, optimizer, loss_func, device, kwargs)
        lr_scheduler.step()
        if epoch % val_freq == 0:
            metric = validate_epoch(model, loader_val, eval_func, device  kwargs)
            print(f'eval = {metric}')
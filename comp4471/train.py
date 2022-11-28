import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from model import ASRID

def train():
    model = ASRID()
    model = model.to(device)

def save():
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

def load():
    model = ASRID()
    model.load_state_dict(torch.load("model.pth"))

# train for single epoch
def train_epoch(model, data_loader, optimizer, loss_func, device, **kwargs):
    # train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, summary_writer, conf, args.local_rank, args.only_changed_frames)

    # Keyword arguments:
    show_every = kwargs.pop('show_every', 100)

    model.train()
    for iter, (X, y) in enumerate(data_loader):
        optimizer.zero_grad(set_to_none=True)
        X = X.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        score = model(X)
        loss = loss_func(score, y).mean()

        loss.backward()
        optimizer.step()

        if iter == 0 or iter % show_every == 0:
            print(f'iter {iter}, loss {loss}')

    #with torch._no_grad():
    #    pass

def validate_epoch(loader):
    model.eval()
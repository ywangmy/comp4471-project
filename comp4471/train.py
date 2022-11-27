import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from model import ASRID

def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ASRID()
    model = model.to(device)

def save():
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

def load():
    model = ASRID()
    model.load_state_dict(torch.load("model.pth"))

def train_epoch(loader):
    # train_epoch(current_epoch, loss_functions, model, optimizer, scheduler, train_data_loader, summary_writer, conf, args.local_rank, args.only_changed_frames)
    pass

    loss = None
    loss.backward()

def validate_epoch(loader):
    pass
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
from data_generation import *
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Use CUDA if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CCANN(nn.Module):
    def __init__(self, conv_size):
        super(CCANN, self).__init__()
        # define each layer here
        self.conv1 = nn.Conv1d(1, 10, conv_size, stride=1)
        self.conv2 =  nn.Conv1d(10, 1, 1, stride = 1)

    def forward(self, x):
      # Define the forward Path of the model here
      # Convolution and pooling layers
        # Convolution Layer 1
        x = F.tanh(self.conv1(x))
        # Convolution Layer 2
        x = self.conv2(x)
        return x

# Instantiate network as net
# Kernel size is lb + rb + 1
def get_network(lb,rb):
    model = CCANN(lb+rb+1)
    model = model.to(device)
    return (model)

def get_optim (model, learn_rate):
    optimizer = torch.optim.SGD(model.parameters(), lr = learn_rate)
    return(optimizer)


def train_net(model, criterion, learn_rate, optimizer, learn_rate_decay, epochs,x,y):
    losses = []
    for e in range(epochs):
        total_loss = 0.0
        optimizer.zero_grad
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        losses.append(total_loss)
        if e%500 == 0:
            print("epoch", e, "loss", total_loss)
            learn_rate = learn_rate*learn_rate_decay
            optimizer = torch.optim.SGD(model.parameters(), lr = learn_rate)
    return(losses)



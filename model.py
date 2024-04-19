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
        self.conv1 = nn.Conv1d(1, 2, 7, stride=1)
        self.conv2 = nn.Conv1d(2, 4, 7, stride=1)
        self.conv3 = nn.Conv1d(4, 8, 7, stride=1)
        self.conv4 = nn.Conv1d(8, 4, 7, stride=1)
        self.conv5 = nn.Conv1d(4, 2, 7, stride=1)
        self.conv6 =  nn.Conv1d(2, 1, 1, stride = 1)

    def forward(self, x):
      # Define the forward Path of the model here
      # Convolution and pooling layers
        # Convolution Layer 1
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        # Convolution Layer 2
        x = self.conv6(x)
        return x

# Instantiate network as net
# Kernel size is lb + rb + 1
def get_network(lb,rb):
    model = CCANN(lb+rb+1)
    
    # Initialize weights
    # nn.init.uniform_(model.conv1.weight, a= -.001, b= .001)
    # nn.init.uniform_(model.conv2.weight, a= -.001, b= .001)
    
    model = model.to(device)
    return (model)

def get_optim (model, learn_rate):
    optimizer = torch.optim.SGD(model.parameters(), lr = learn_rate, weight_decay=.01)#10^-3 seems best
    return(optimizer)


def train_net(model, criterion, learn_rate, optimizer, learn_rate_decay, epochs, Data):
    losses = []
    for e in range(epochs):
        total_loss = 0.0
        for batch_idx,batch in enumerate(Data):
            x=batch[0].to(device)
            y=batch[1].to(device)
            # print("Size x" + str(x.size()))
            # print("Size y" + str(y.size()))

            optimizer.zero_grad
            with torch.set_grad_enabled(True):
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        losses.append(total_loss / (batch_idx + 1))
        if e%500 == 0:
            print("epoch", e, "loss", total_loss)
            learn_rate = learn_rate*learn_rate_decay
            optimizer = torch.optim.SGD(model.parameters(), lr = learn_rate)
    return(losses)



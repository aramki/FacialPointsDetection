## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Sequential(nn.Conv2d(1,16,3,padding=0), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2, stride=2))
        #output = 222x222

        self.conv2 = nn.Sequential(nn.Conv2d(16,32,3, padding=1),nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,stride=2))
        #output = 111x111

        self.conv3 = nn.Sequential(nn.Conv2d(32,64,4, padding=1),nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, stride=2))
        #output = 56x56

        self.conv4 = nn.Sequential(nn.Conv2d(64,64,3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),nn.MaxPool2d(2, stride=2))
        #output = 28x28
        
        self.conv5 = nn.Sequential(nn.Conv2d(64,64,4, padding=1), nn.BatchNorm2d(64), nn.ReLU(),nn.MaxPool2d(2, stride=2))
        #output = 13x13
        
        self.lr = nn.Sequential(nn.Dropout(p=0.4),nn.Linear(13*13*64,136))       
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        network = self.conv1(x)
        network = self.conv2(network)
        network = self.conv3(network)        
        network = self.conv4(network)
        network = network.view(x.size(0),-1)
        network = self.lr(network)

        return network
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

### AlexNet
class AlexNet(nn.Module):
  def __init__(self, in_channels, out_channels, config=None):
    super(AlexNet, self).__init__()

    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

    self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
    self.fc2 = nn.Linear(in_features=120, out_features=60)
    self.out = nn.Linear(in_features=60, out_features=out_channels)

  def forward(self, t):
    # conv 1
    t = self.conv1(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2)

    # conv 2
    t = self.conv2(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2)

    # fc1
    t = t.reshape(-1, 12*4*4) # x.view - оставила коммент на погуглить
    t = self.fc1(t)
    t = F.relu(t)

    # fc2
    t = self.fc2(t)
    t = F.relu(t)

    # output
    t = self.out(t)
    # don't need softmax here since we'll use cross-entropy as activation.

    return t

### VGG
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        
    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2)        
        return t
    
    
class TripleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TripleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)       
        
    def forward(self, t):
        t = self.conv1(t)
        t = F.relu(t)
        t = self.conv2(t)
        t = F.relu(t)
        t = self.conv3(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2)        
        return t
        
class VGGBlock16(nn.Module):
    def __init__(self,  in_channels, out_channels, height, width, config=None):
        super(VGGBlock16, self).__init__()
        
        self.double_conv1 = DoubleConv(in_channels, 8)
        self.double_conv2 = DoubleConv(8, 16)
        
        self.triple_conv1 = TripleConv(16, 32)
        self.triple_conv2 = TripleConv(32, 64)
        self.triple_conv3 = TripleConv(64, 64)
        
        self.fc1 = nn.Linear(in_features=math.ceil((height/32)*(width/32))*64, out_features=60)
        self.fc2 = nn.Linear(in_features=60, out_features=60)
        self.fc3 = nn.Linear(in_features=60, out_features=out_channels)
        
    def forward(self, t):
        # 2 двойных свертки:
        t = self.double_conv1(t)
        t = self.double_conv2(t)
        
        # 3 тройных свертки:
        t = self.triple_conv1(t)
        t = self.triple_conv2(t)
        t = self.triple_conv3(t)
        
        # 3 полносвязных слоя
        t = torch.flatten(t)
        t = self.fc1(t)
        t = F.relu(t)
        t = self.fc2(t)
        t = F.relu(t)
        t = self.fc3(t)
        #softmax
        return t
        
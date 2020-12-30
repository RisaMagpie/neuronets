import torch
import torch.nn as nn
import torch.nn.functional as F
import math

### AlexNet
class AlexNet(nn.Module):
  def __init__(self, in_channels, out_channels, config=None):
    super().__init__()

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
    t = t.reshape(-1, t.shape[1]) # x.view - оставила коммент на погуглить
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

class xConv(nn.Module):
    def __init__(self, num_of_conv_layers, in_channels, out_channels, **kwargs):
        super().__init__()
        self.num_of_conv_layers = num_of_conv_layers
        setattr(self, 'conv0', 
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=out_channels, 
                          kernel_size=3, padding=1, **kwargs))
        for i in range(1, num_of_conv_layers):
            setattr(self, 'conv'+str(i), 
                    nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels, 
                              kernel_size=3, padding=1, **kwargs))
        
    def forward(self, t):
        for i in range(self.num_of_conv_layers):
            t = getattr(self, 'conv'+str(i))(t)
            t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2)  
        return t
        
class VGG16(nn.Module):
    def __init__(self,  in_channels, out_channels):
        super().__init__()
        self.num_of_fc_layers = 3

        self.double_conv1 = xConv(2, in_channels, 8)
        self.double_conv2 = xConv(2, 8, 16)
        
        self.triple_conv1 = xConv(3, 16, 32)
        self.triple_conv2 = xConv(3, 32, 64)
        self.triple_conv3 = xConv(3, 64, 64)

        
        for num, in_features, out_features in zip(
            range(self.num_of_fc_layers), [64,60,60],[60,60,out_channels]
        ):
            setattr(self, 'fc'+str(num),
                   nn.Linear(in_features=in_features, out_features=out_features))
        
    def forward(self, t):
        # 2 двойных свертки:
        t = self.double_conv1(t)
        t = self.double_conv2(t)
        
        # 3 тройных свертки:
        t = self.triple_conv1(t)
        t = self.triple_conv2(t)
        t = self.triple_conv3(t)
        
        if t.shape[2]>1 or t.shape[3]>1:
            t = nn.AdaptiveAvgPool2d((1,1))(t)
  
        # 3 полносвязных слоя
        t = t.reshape(-1, t.shape[1])
        for i in range(self.num_of_fc_layers):
            t = getattr(self, 'fc'+str(i))(t)
            if i<self.num_of_fc_layers:
                t = F.relu(t)

        #softmax
        return t
        
        
### ResNet

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.conv_layers_num = 3
        super().__init__()
        for num, (kern_size, padd_size) in enumerate(zip([1,3], [0,1])):
            setattr(self, 'conv'+str(num),
                    nn.Conv2d(in_channels=in_channels, 
                              out_channels=in_channels, 
                              kernel_size=kern_size, padding=padd_size))
        setattr(self, 'conv'+str(self.conv_layers_num-1),
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=out_channels, 
                          kernel_size=kern_size, padding=padd_size))
        self.skip = None
        if in_channels!=out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
        
    def forward(self, t):
        ident = t
        for i in range(self.conv_layers_num):
            t = getattr(self, 'conv'+str(i))(t)
            if i<self.conv_layers_num:
                t=F.relu(t)
        if self.skip:
            ident = self.skip(ident)
        t = t + ident
        t=F.relu(t)
        return t
                
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride = 2)
        self.resblocks = nn.ModuleList([ResBlock(64*i, 64*(i+1)) for i in range(1,3)])
        self.fc1 = nn.Linear(in_features=384, out_features=out_channels)     
        
    def forward(self, t):                
        t = self.conv1(t)
        t = F.max_pool2d(t, kernel_size=3, stride=2)  
        for block in self.resblocks:
            t = block(t)
    
        if t.shape[2]>1 or t.shape[3]>1:
            t = nn.AdaptiveAvgPool2d((1,1))(t)

        t = t.reshape(-1, t.shape[1]*t.shape[0])
        t = self.fc1(t)
        return t
        
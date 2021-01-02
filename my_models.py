import torch
import torch.nn as nn
import torch.nn.functional as F
import json

### AlexNet
class AlexNet(nn.Module):
  def __init__(self, in_channels, out_channels, config=None):
    super().__init__()

    self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

    self.fc1 = nn.Linear(in_features=300, out_features=120)
    self.fc2 = nn.Linear(in_features=120, out_features=60)
    self.out = nn.Linear(in_features=60, out_features=out_channels)

  def forward(self, tensor):
    # conv 1
    tensor = self.conv1(tensor)
    tensor = F.relu(tensor)
    tensor = F.max_pool2d(tensor, kernel_size=2)

    # conv 2
    tensor = self.conv2(tensor)
    tensor = F.relu(tensor)
    tensor = F.max_pool2d(tensor, kernel_size=2)

    # fc1
    tensor = torch.flatten(tensor, 1)
    tensor = self.fc1(tensor)
    tensor = F.relu(tensor)

    # fc2
    tensor = self.fc2(tensor)
    tensor = F.relu(tensor)

    # output
    tensor = self.out(tensor)
    # don't need softmax here since we'll use cross-entropy as activation.

    return tensor

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
        
    def forward(self, tensor):
        for i in range(self.num_of_conv_layers):
            tensor = getattr(self, 'conv'+str(i))(tensor)
            tensor = F.relu(tensor)
        tensor = F.max_pool2d(tensor, kernel_size=2)  
        return tensor
        
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
        
    def forward(self, tensor):
        # 2 двойных свертки:
        tensor = self.double_conv1(tensor)
        tensor = self.double_conv2(tensor)
        
        # 3 тройных свертки:
        tensor = self.triple_conv1(tensor)
        tensor = self.triple_conv2(tensor)
        tensor = self.triple_conv3(tensor)
        
        if tensor.shape[2]>1 or tensor.shape[3]>1:
            tensor = nn.AdaptiveAvgPool2d((1,1))(tensor)
  
        # 3 полносвязных слоя
        tensor = torch.flatten(tensor, 1)
        for i in range(self.num_of_fc_layers):
            tensor = getattr(self, 'fc'+str(i))(tensor)
            if i<self.num_of_fc_layers:
                tensor = F.relu(tensor)

        #softmax
        return tensor
        
        
### ResNet

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layers_num = 3
        self.conv0 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=in_channels, 
                               kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=in_channels, 
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=1, padding=0)

        if in_channels!=out_channels:
            self.adjust_skip_size = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))
        else:
            self.adjust_skip_size = None
        
    def forward(self, tensor):
        input_tensor = tensor
        for i in range(self.conv_layers_num):
            tensor = getattr(self, 'conv'+str(i))(tensor)
            if i<self.conv_layers_num:
                tensor = F.relu(tensor)
                
        if self.adjust_skip_size:
            input_tensor = self.adjust_skip_size(input_tensor)
        tensor = tensor + input_tensor
        tensor = F.relu(tensor)
        return tensor
                
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride = 2)
        self.resblocks = nn.ModuleList([ResBlock(64*i, 64*(i+1)) for i in range(1,3)])
        self.fc1 = nn.Linear(in_features=192, out_features=out_channels)     
        
    def forward(self, tensor):                
        tensor = self.conv1(tensor)
        tensor = F.max_pool2d(tensor, kernel_size=3, stride=2)  
        for block in self.resblocks:
            tensor = block(tensor)
    
        if tensor.shape[2]>1 or tensor.shape[3]>1:
            tensor = nn.AdaptiveAvgPool2d((1,1))(tensor)

        tensor = torch.flatten(tensor, 1)
        tensor = self.fc1(tensor)
        return tensor

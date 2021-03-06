import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import typing

### AlexNet
class AlexNet(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 *args, **kwargs):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=6, 
                               kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, 
                               out_channels=12, 
                               kernel_size=5)

        self.fc1 = nn.Linear(in_features=300, 
                             out_features=120)
        self.fc2 = nn.Linear(in_features=120, 
                             out_features=60)
        self.out = nn.Linear(in_features=60, 
                             out_features=out_channels)

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
    def __init__(self, 
                 num_of_conv_layers: int, 
                 in_channels: int, 
                 out_channels: int):
        super().__init__()
        self.num_of_conv_layers = num_of_conv_layers
        setattr(self, 'conv0', 
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=out_channels, 
                          kernel_size=3, padding=1))
        for i in range(1, num_of_conv_layers):
            setattr(self, 'conv'+str(i), 
                    nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels, 
                              kernel_size=3, padding=1))
        
    def forward(self, tensor):
        for i in range(self.num_of_conv_layers):
            tensor = getattr(self, 'conv'+str(i))(tensor)
            tensor = F.relu(tensor)
        tensor = F.max_pool2d(tensor, kernel_size=2)  
        return tensor
    
class VGG(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 conv_blocks_out_size: typing.List[int], 
                 conv_blocks_amounts: typing.List[int],
                 linear_layers_out_size: typing.List[int], # last element must be equals to out_channels
                 *args, **kwargs):
        super().__init__()
        
        # Convolution layers
        self.xconv_layers = nn.ModuleList([])
        conv_blocks_out_size.insert(0, in_channels) # добавление input channnels для первого блока
        for (layer_depth, in_channels_iterator, out_channels_iterator) in zip(conv_blocks_amounts,  conv_blocks_out_size[:-1], conv_blocks_out_size[1:]):            
            self.xconv_layers = self.xconv_layers.append(
                xConv(layer_depth, in_channels_iterator, out_channels_iterator) 
            )

        # Linear layers:
        self.linear_layers = nn.ModuleList([])
        linear_layers_out_size.insert(0, conv_blocks_out_size[-1])
        for (in_channels_iterator, out_channels_iterator) in zip(
             linear_layers_out_size[:-1], linear_layers_out_size[1:]
        ):
            self.linear_layers = self.linear_layers.append(
                nn.Linear(in_features=in_channels_iterator, 
                          out_features=out_channels_iterator)
            )
            
    def forward(self, tensor):
        for layer in self.xconv_layers:
            tensor = layer(tensor)
        tensor = nn.AdaptiveAvgPool2d((1,1))(tensor) 
        tensor = torch.flatten(tensor, 1)
        for layer in self.linear_layers[:-1]:
            tensor = layer(tensor)
            tensor = F.relu(tensor)
        tensor = self.linear_layers[-1](tensor)
        return tensor


        
        
        
### ResNet       

class ResBlock(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 is_bottleneck_block: bool):
        super().__init__()
        
        if is_bottleneck_block:
            self.res_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=in_channels//4, 
                          kernel_size=1, padding=0),
                nn.BatchNorm2d(num_features=in_channels//4),
                nn.ReLU(),

                nn.Conv2d(in_channels=in_channels//4, 
                          out_channels=in_channels//4, 
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=in_channels//4),
                nn.ReLU(),

                nn.Conv2d(in_channels=in_channels//4, 
                          out_channels=out_channels, 
                          kernel_size=1, padding=0), 
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.res_block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=in_channels,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(),

                nn.Conv2d(in_channels=in_channels, 
                          out_channels=out_channels,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=out_channels)
            )

        if in_channels!=out_channels:
            self.adjust_skip_size = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=out_channels, 
                          kernel_size=1),
                nn.BatchNorm2d(num_features=out_channels)
            )
        else:
            self.adjust_skip_size = None
        
    def forward(self, tensor):
        input_tensor = tensor
        tensor = self.res_block(tensor)
        if self.adjust_skip_size:
            input_tensor = self.adjust_skip_size(input_tensor)
        tensor = tensor + input_tensor
        tensor = F.relu(tensor)
        return tensor
    
class ResNetLayer(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 layer_depth: int, 
                 is_bottleneck_blocks: bool):
        super().__init__()
        self.resblocks = nn.ModuleList([
            ResBlock(in_channels, out_channels, is_bottleneck_blocks)
        ])
        self.resblocks = self.resblocks.extend(nn.ModuleList([
            ResBlock(out_channels, out_channels, is_bottleneck_blocks) 
             for block_type in range(layer_depth-1)
        ]))
        
    def forward(self, tensor):
        for block in self.resblocks:
            tensor = block(tensor)
        return tensor
        
                
class ResNet(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int, 
                 blocks_out_size: typing.List[int], 
                 blocks_amounts: typing.List[int],
                 is_bottleneck_blocks: bool = False,
                 *args, **kwargs):
        super().__init__()
        
        self.conv0 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=64, 
                               kernel_size=3, 
                               stride = 2)
        self.batch_norm0 = nn.BatchNorm2d(num_features=64)
        
        blocks_out_size.insert(0, 64) # добавление input channnels для первого ResBlock'a
        self.reslayers = nn.ModuleList([])
        
        for (
            layer_depth, in_channels_iterator, out_channels_iterator
        ) in zip(
            blocks_amounts, blocks_out_size[:-1], blocks_out_size[1:]
        ): 
            self.reslayers = self.reslayers.append(
                ResNetLayer(in_channels_iterator, out_channels_iterator, layer_depth, is_bottleneck_blocks) 
            )        
        self.fc0 = nn.Linear(in_features=blocks_out_size[-1], out_features=out_channels)     
        
    def forward(self, tensor):                
        tensor = self.conv0(tensor)
        tensor = self.batch_norm0(tensor)
        tensor = F.relu(tensor)
        tensor = F.max_pool2d(tensor, kernel_size=3, stride=2)  
        
        for layer in self.reslayers:
            tensor = layer(tensor)

        tensor = nn.AdaptiveAvgPool2d((1,1))(tensor)
        tensor = torch.flatten(tensor, 1)
        
        tensor = self.fc0(tensor)
        return tensor

    
    
### UNet

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_depth=4):
        super().__init__()   
        self.downsample_depth = downsample_depth
        
        # downsample     
        self.downsample = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels*pow(2, i), out_channels=in_channels*pow(2, i), kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=in_channels*pow(2, i)),
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels*pow(2, i), out_channels=in_channels*pow(2, i+1), kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=in_channels*pow(2, i+1)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            for i in range(downsample_depth)
        ])
        
        # inner part
        self.inner_part = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*pow(2, downsample_depth), 
                               out_channels=in_channels*pow(2, downsample_depth), 
                               kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=in_channels*pow(2, downsample_depth)),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels*pow(2, downsample_depth), 
                               out_channels=in_channels*pow(2, downsample_depth), 
                               kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=in_channels*pow(2, downsample_depth)),
            nn.ReLU()
        
        )
        
        # upsample
        self.upsamples = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels*pow(2, i), 
                                           out_channels=in_channels*pow(2, i-1),
                                           kernel_size=2, stride=2),
                nn.BatchNorm2d(num_features=in_channels*pow(2, i-1)),
                nn.ReLU()
            )
            for i in range(downsample_depth, 0, -1)
        ])
        
        self.conv_upsample_with_changes_ch_amount = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels*pow(2, i), 
                      out_channels=in_channels*pow(2, i-1), 
                      kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=in_channels*pow(2, i-1)),
                nn.ReLU()
            )
            for i in range(downsample_depth, 0, -1)
        ])
        self.conv_upsample_without_changes_ch_amount = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels*pow(2, i-1), 
                      out_channels=in_channels*pow(2, i-1), 
                      kernel_size=3, padding=1),
                nn.BatchNorm2d(num_features=in_channels*pow(2, i-1)),
                nn.ReLU()
            )
            for i in range(downsample_depth, 0, -1)
        ])
        self.last_conv = nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=1)

        
    def forward(self, tensor):        
        saved_tensors = []
        
        # downsample
        for down_block in self.downsample:
            saved_tensors.append(tensor)
            tensor = down_block(tensor)
            
        # inner part  
        tensor = self.inner_part(tensor)

        # upsample
        for block_num, up_block in enumerate(self.upsamples):
            tensor = up_block(tensor)
            tensor = torch.cat((tensor, saved_tensors[self.downsample_depth-block_num-1]), 1)
            tensor = self.conv_upsample_with_changes_ch_amount[block_num](tensor)
            tensor = self.conv_upsample_without_changes_ch_amount[block_num](tensor)
        tensor = self.last_conv(tensor)
        tensor = nn.Sigmoid()(tensor)
        return tensor
    
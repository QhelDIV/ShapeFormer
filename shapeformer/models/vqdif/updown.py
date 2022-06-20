'''
Code from the 3D UNet implementation:
https://github.com/wolny/pytorch-3dunet/
'''
import importlib
import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial

def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


def conv3d(in_channels, out_channels, kernel_size, bias, stride=1, padding=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, stride=1, padding=1):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int): add zero-padding to the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, stride=stride, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class ConvLayer(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='crg', num_groups=8, stride=1, padding=1):
        super().__init__()
        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, stride=stride, padding=padding):
            self.add_module(name, module)

class Downsampler(nn.Module):
    """Keep the receptive field as small as possible

    Args:
        nn ([type]): [description]
    """
    def __init__(self, in_channels, downsample_steps=1):
        super().__init__()
        channels = [in_channels * (2 ** k) for k in range(0, downsample_steps+1)]
        self.blocks = []
        for i in range(downsample_steps):
            in_c, out_c = channels[i], channels[i+1]
            self.blocks.append( ConvLayer(in_channels=in_c,  out_channels=out_c, kernel_size=2, order="crg", stride=2, padding=0) )
            self.blocks.append( ConvLayer(in_channels=out_c, out_channels=out_c, kernel_size=1, order="crg", stride=1, padding=0) )
        self.blocks = nn.Sequential(*self.blocks)
        #self.blocks.append( ConvLayer(in_channels=channels[-1], out_channels=channels[-1], kernel_size=1, corder="c", stride=1, padding=0) )
    def forward(self, x):
        return self.blocks(x)
class Upsampler(nn.Module):
    def __init__(self, in_channels, upsampler_steps=1, mode="nearest"):
        super().__init__()
        channels = [int( in_channels / (2 ** k) ) for k in range(0, upsampler_steps+1)]
        self.blocks = []
        for i in range(upsampler_steps):
            in_c, out_c = channels[i], channels[i+1]
            self.blocks.append( nn.Upsample(scale_factor=2, mode=mode) )#, align_corners=False), )
            self.blocks.append( ConvLayer(in_channels=in_c,  out_channels=out_c, kernel_size=3, order="crg", stride=1, padding=1) )
            self.blocks.append( ConvLayer(in_channels=out_c, out_channels=out_c, kernel_size=3, order="crg", stride=1, padding=1) )
        self.blocks = nn.Sequential(*self.blocks)
        #self.blocks.append( ConvLayer(in_channels=channels[-1], out_channels=channels[-1], kernel_size=1, corder="c", stride=1, padding=0) )
    def forward(self, x):
        return self.blocks.forward(x)

if __name__ == "__main__":
    """
    testing
    """
    in_channels = 1
    out_channels = 1
    f_maps = 32
    num_levels = 3
    model = UNet3D(in_channels, out_channels, f_maps=f_maps, num_levels=num_levels, layer_order='cr')
    print(model)

    reso = 42
    
    import numpy as np
    import torch
    x = np.zeros((1, 1, reso, reso, reso))
    x[:,:, int(reso/2-1), int(reso/2-1), int(reso/2-1)] = np.nan
    x = torch.FloatTensor(x)

    out = model(x)
    print('%f'%(torch.sum(torch.isnan(out)).detach().cpu().numpy()/(reso*reso*reso)))
    
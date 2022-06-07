import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ResnetBlockFC
from .common import normalize_coordinate, normalize_3d_coordinate, map2local

from .unet3d import UNet3D
from .updown import Upsampler

class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128, unet3d=False, unet3d_kwargs=None, upsampler=False, upsampler_kwargs=None,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        if upsampler:
            self.upsampler = Upsampler(**upsampler_kwargs)
        else:
            self.upsampler = None

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])


        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding
    

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c


    def forward(self, p, c_grid, **kwargs):
        # unet & upsample
        # in & out: (B, k*C, res/k, res/k, res/k), k=2**downsample_steps

        if self.unet3d is not None:
            uneted    = self.unet3d(c_grid)
        else:
            uneted    = c_grid
        # (B, C, res,res,res)
        if self.upsampler is not None:
            upsampled = self.upsampler(uneted)
        else:
            upsampled = uneted
        # implicit decoder
        c = self.sample_grid_feature(p, upsampled)
        c = c.transpose(1, 2)

        p = p.float()
        net = self.fc_p(p)

        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)

        out = self.fc_out(self.actvn(net))
        out = out

        return out


import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ResnetBlockFC
from torch_scatter import scatter_mean, scatter_max
from .common import coordinate2index, normalize_coordinate, normalize_3d_coordinate, map2local
#from .encoder.unet import UNet
from .unet3d import UNet3D
from .updown import Downsampler

class LocalPoolPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', downsampler=False, downsampler_kwargs=None,
                    c2i_order="original", grid_resolution=None, plane_type='grid', padding=0.1, n_blocks=5):
        super().__init__()
        self.c_dim = c_dim
        self.c2i_order = c2i_order
        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        # if unet3d:
        #     self.unet3d = UNet3D(**unet3d_kwargs)
        # else:
        #     self.unet3d = None
        if downsampler:
            self.downsampler = Downsampler(**downsampler_kwargs)
        else:
            self.downsampler = None

        #self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

    def generate_grid_features(self, p, c):
        self.p_nor = p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d', c2i_order=self.c2i_order)
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x C x reso^3
        # (B, C, res, res, res)
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid) # sparse matrix (B x 512 x reso x reso)
        self.pc_feature_grid = fea_grid.detach().clone()

        #if self.unet3d is not None:
        #    fea_grid = self.unet3d(fea_grid)
        if self.downsampler is not None:
            # produce (B, k*C, res/k, res/k, res/k), k=2**downsample_steps
            fea_grid = self.downsampler(fea_grid)
           
        out_reso_grid = fea_grid.shape[-1]
        
        mask_ind      = (p_nor * out_reso_grid).long() #coordinate2index(p_nor, out_reso_grid, coord_type='3d', c2i_order=self.c2i_order)
        #print(mask_ind.shape, index.shape, self.reso_grid, out_reso_grid)
        #print( len(torch.unique(mask_ind, axis=-1)) )
        inds_flat     = mask_ind.view(-1, mask_ind.shape[-1])
        binds = torch.repeat_interleave(torch.arange(mask_ind.shape[0]).type_as(mask_ind).long(), mask_ind.shape[1])
        mask = torch.zeros(p.shape[0], out_reso_grid, out_reso_grid, out_reso_grid).bool()
        mask[binds, inds_flat[:,2], inds_flat[:,1], inds_flat[:,0]] = True

        return fea_grid, mask

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid**3)
            else:
                raise NotImplementedError()
                #fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane**2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)


    def forward(self, p):
        batch_size, T, D = p.size()

        # acquire the index for each point
        coord = {}
        index = {}
        if 'grid' in self.plane_type:
            coord['grid'] = normalize_3d_coordinate(p.clone(), padding=self.padding)
            index['grid'] = coordinate2index(coord['grid'], self.reso_grid, coord_type='3d', c2i_order=self.c2i_order)
        
        net = self.fc_pos(p)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea, mask = self.generate_grid_features(p, c)
        
        #sparse_fea = self.dense2sparse(p, fea)

        # return: (B, k*C, res/k, res/k, res/k), k=2**downsample_steps
        return fea, mask

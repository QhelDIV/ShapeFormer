
import numpy as np
from torch.utils.data import DataLoader, Dataset
import glob
import os
from pathlib import Path
from xgutils import *
import torch

"""
for cmp_hprscan
[793, 637, 604, 594, 496]
[41,179,188]

for 

[[3, 4, 16, 18, 20, 22], [3, 12, 14, 15, 17], [62, 80, 82,100,0,10,20,30,40,50]]
"""
from scipy.spatial.transform import Rotation as R
def apply_random_rotation(points):
    r = R.random()
    return r.apply(points), r
def apply_random_rotation_axis(points, axis=1):
    dim = points.shape[-1]
    zhou = np.zeros(dim)
    zhou[axis] = 1
    random_angle = np.random.rand()*2*np.pi
    r = R.from_rotvec( random_angle * zhou)
    rotated = r.apply(points)
    return rotated, r
def apply_random_scaling(points, max_bound=0.99):
    extent = np.abs(points).max()
    max_ratio = max_bound / extent
    scaling = 1 + (np.random.rand(1)[0] * (max_ratio-1) )
    scaled = scaling * points
    return scaled, scaling

def apply_random_shift(points, max_bound=0.99):
    dim = points.shape[-1]
    hbd, lbd = points.max(axis=0), points.min(axis=0)
    leng = hbd-lbd
    hshift, lshift = 1-hbd, -1-lbd
    shift = np.random.rand(1, dim) * (hshift-lshift) + lshift
    shifted = points + shift
    return shifted, shift
def apply_random_transforms(X, Ys={}, mode=[], max_voxels=812, voxel_dim=16):
    hbd, lbd = X.max(axis=0), X.min(axis=0)
    center = (hbd+lbd)/2
    leng = hbd-lbd
    
    Xbd2 = (X - center)/leng.max() * .6         # fit to [-.6, .6]
    for key in Ys:
        Ys[key] = (Ys[key]-center)/leng.max() * .6
    
    if "rot_axis_y" in mode:
        Xbd2, r = apply_random_rotation_axis(Xbd2, axis=1)
        for key in Ys:
            Ys[key] = r.apply(Ys[key])
    if "rot" in mode:
        Xbd2, r = apply_random_rotation(Xbd2)
        for key in Ys:
            Ys[key] = r.apply(Ys[key])
    if "scale" in mode:
        Xbd2, scaling = apply_random_scaling(Xbd2)
        for key in Ys:
            Ys[key] = Ys[key] * scaling

    voxel = ptutil.ths2nps( ptutil.point2voxel( torch.from_numpy(Xbd2[None,...]), grid_dim=voxel_dim) ) 
    voxelN= voxel.sum().astype(np.float32)
    if voxelN > max_voxels:
        safe_scale = max_voxels / voxelN
        safe_scale = safe_scale**(2/3.)
        Xbd2 *= safe_scale
        for key in Ys:
            Ys[key] = Ys[key] * safe_scale
        nvoxel = ptutil.ths2nps( ptutil.point2voxel( torch.from_numpy(Ys["Xbd"][None,...]), grid_dim=voxel_dim) ) 
        nvoxelN= nvoxel.sum()

    if "shift" in mode:
        Xbd2, shift = apply_random_shift(Xbd2)
        for key in Ys:
            Ys[key] = Ys[key] + shift        
    return Ys

class TransformDataset(Dataset):
    def __init__(self, split="test", mode=["rot_axis_y", "scale"], apply_Xtg=False, 
                    max_voxels=100, voxel_dim=16, dset_opt={}):
        """appy random 3d transformations toward points

        Args:
            split (str, optional): [description]. Defaults to "test".
            mode (str, optional): "scale" or "axis". Defaults to "scale".
            dset_opt (dict, optional): [description]. Defaults to {}.
        """
        super().__init__()
        self.__dict__.update(locals())
        self.dset = sysutil.instantiate_from_opt(dset_opt)

    def __len__(self):
        return len(self.dset)
    def __getitem__(self, ind):
        ditem = self.dset[ind]
        if "Xbd" in ditem:
            nditem = {"Xbd":ditem["Xbd"].copy()}
            if "Xct" in ditem:
                nditem["Xct"] = ditem["Xct"].copy()
            if "Xtg" in ditem and self.apply_Xtg==True:
                nditem["Xtg"] = ditem["Xtg"].copy()
            ret = apply_random_transforms(ditem["Xbd"].copy(), nditem, mode=self.mode, max_voxels=self.max_voxels, voxel_dim=self.voxel_dim)
            for key in ret:
                ditem[key] = ret[key].astype(np.float32)
        return ditem

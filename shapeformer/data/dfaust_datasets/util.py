
import torch
import sklearn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from xgutils import *
import scipy
import h5py
import time
import glob
import igl
import os
import traceback

#resolution = gt
version = f"v1"
def apply_transform( Xbd ):
    shift = (Xbd.max(axis=0) + Xbd.min(axis=0))/2
    Xbd -= shift[None,...]
    return Xbd
def generate_gt_samples(shape_path, sample_N=64**3, near_std=0.015, far_std=0.2):
    vert, face = igl.read_triangle_mesh(shape_path)
    vert = apply_transform(vert)
    if np.abs(vert).max()>1.:
        print("Warning, data exceeds bbox 1.", shape_path, np.abs(vert).max())
    Xbd  = geoutil.sampleMesh(vert, face, sample_N)
    
    near_num = sample_N // 2
    far_num  = sample_N - near_num

    near_pts = Xbd[:near_num].copy()
    far_pts  = Xbd[near_num:].copy()

    near_pts += near_std * np.random.randn(*near_pts.shape) 
    far_pts  += far_std  * np.random.randn(*far_pts.shape)
    
    Xtg = np.concatenate([near_pts, far_pts], axis=0)
    mask = np.logical_or(Xtg > .99 , Xtg < -.99)
    Xtg[mask] = np.random.rand(mask.sum())*2 - 1
    Xtg = Xtg.clip(-.99, .99)
    assert Xtg.min()>=-1.00001 and Xtg.max()<=1.00001
    Ytg, _, _ = geoutil.signed_distance(Xtg, vert, face)

    Xtg = Xtg.astype(np.float16)
    Ytg = Ytg.astype(np.float16)
    Xbd = Xbd.astype(np.float16)
    return Xbd, Xtg, Ytg
    
def generate_dataitem(shape_path, selem_size=3):
    vert, face = igl.read_triangle_mesh(shape_path)
    vert = apply_transform(vert)
    if np.abs(vert).max()>1.:
        print("Warning, data exceeds bbox 1.", shape_path, np.abs(vert).max())
    Ytg = geoutil.mesh2sdf(vert, face, gridDim=resolution)[...,3]
    # voxel, _ = geoutil.morph_voxelization(vert, face, grid_dim=256, selem_size=selem_size)
    
    # shape_vocab, vocab_idx = ptutil.compress_voxels(voxel, packbits=True)
    
    # v256, f256 = geoutil.array2mesh(voxel.reshape(-1), dim=3, bbox=np.array([[-1,-1,-1],[1,1,1.]])*1., thresh=.5, if_decimate=False, cart_coord=True)
    Xbd  = geoutil.sampleMesh(vert, face, 65536)
    
    return Xbd, Ytg
def voxelize_dfaust_shape(shape_path, selem_size=0):
    try:
        shape_pn   = ".".join(shape_path.split(".")[:-1])
        #out_path   = shape_pn + f"_{version}.npy"
        #shape_vocab, vocab_idx, Xbd = generate_dataitem(shape_path, selem_size=selem_size)
        #data = dict(shape_vocab=shape_vocab, vocab_idx=vocab_idx, Xbd=Xbd)
        #loaded = np.load( out_path, allow_pickle=True).item()
        Xbd, Xtg, Ytg = generate_gt_samples(shape_path)
        np.save( shape_pn + f"_{version}_Xbd.npy", Xbd)
        np.save( shape_pn + f"_{version}_Xtg.npy", Xtg)
        np.save( shape_pn + f"_{version}_Ytg.npy", Ytg)
        #Xbd, Ytg = generate_dataitem(shape_path, selem_size=selem_size)
        #data = dict(Xbd=Xbd, Ytg=Ytg)
        #np.save( out_path, data)
    except Exception as e:
        traceback.print_exc()
        print(f'Error encountered during voxelization:{shape_path}', e)
        return 1
    return 0

def voxelize_dfaust(data_root="/studio/datasets/DFAUST/data/"):
    shapes = glob.glob( os.path.join(data_root, "*/*.obj") )
    #shapes = shapes[:40]
    print("num of shapes", len(shapes))
    #print(shapes)
    #for shape_dir in sysutil.progbar(shape_dirs):
    #    print(shape_dir)
    #    voxelize_partnet_shape(shape_dir)
    
    return_codes = sysutil.parallelMap(voxelize_dfaust_shape, [shapes], zippedIn=False)
    np.save(f"/studio/datasets/DFAUST/voxelization_failure_code.npy",return_codes)
    print("Percentage of failure:", np.array(return_codes).sum()/len(shapes))
    print("Return code:", return_codes)

"""
Contrary to PartNet_datasets.py
These datasets shrink the partnet pc by .9 in order to make the mesh fall into [-1,1] bounding box.
Though there are still some shapes violate this principle, most of them satisfy the rule.
Also, in these datasets Ytg (ground truth voxel value) are available
"""
import torch
import sklearn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from xgutils import *
import scipy
import h5py
import time
import os
import random
import trimesh
import csv
import json
import pickle
#from util.pc_utils import rotate_point_cloud_by_axis_angle, sample_point_cloud_by_n
import glob
import igl
#SPLIT_DIR = "/studio/Multimodal-Shape-Completion/data/partnet_train_val_test_split"
PARTNET_DIR = "/studio/datasets/PartNet/"
class RealScanDataset(Dataset):
    def __init__(self, split="test", cate="Chair", sample_N=8192, category=None, **kwargs):
        print(kwargs)
        self.__dict__.update(locals())
        super().__init__()
        if self.category is not None:
            self.cate = cate = self.category

        self.data_dir = f"/studio/nnrecon/datasets/PartNet/eval_dset/{cate}/"
        self.length = len(glob.glob(self.data_dir+"*.npy"))//2
        if cate=="Car":
            data_dir = "/studio/nnrecon/datasets/pcl2pcl/kitti_val/"
            objs = glob.glob(data_dir+"*.ply")
        if cate=="Chair":
            data_dir = "/studio/nnrecon/datasets/pcl2pcl/scannet_v2_chairs_aligned/"
        if cate=="Table":
            data_dir = "/studio/nnrecon/datasets/pcl2pcl/scannet_v2_tables_aligned/"
        if cate=="Chair" or cate=="Table":
            fp = data_dir + "point_cloud_test_split.pickle"
            with open(fp,"rb") as f:
                objs = pickle.load(f)
            for i in range(len(objs)):
                objs[i] = data_dir+"point_cloud/"+objs[i]
        self.length = len(objs)
        self.objs = objs
        print("dataset length", self.length)
    def __getitem__(self, index):
        shapep = self.objs[index]
        vert, face = igl.read_triangle_mesh(shapep)
        if self.cate == "Car":
            vert*=2
        else:
            vert -= (vert.max(axis=0)+vert.min(axis=0))/2.
            # if self.cate=="Chair":
            #     vert *= 1.2
            # else:
            #     vert *= .7
            vert = vert/np.abs(vert).max() * .6
            #vert= vert/3.
        print(vert.max(axis=0))
        vert[:,2] = -vert[:,2]
        choice = np.random.choice(vert.shape[0], self.sample_N)

        return {"Xct":vert[choice].astype(np.float32)}
    def __len__(self):
        return self.length
    
    
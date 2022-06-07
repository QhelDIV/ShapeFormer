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
#from util.pc_utils import rotate_point_cloud_by_axis_angle, sample_point_cloud_by_n
import glob
#SPLIT_DIR = "/studio/Multimodal-Shape-Completion/data/partnet_train_val_test_split"
PARTNET_DIR = "/studio/datasets/PartNet/"
class EvalPartNetDataset(Dataset):
    def __init__(self, split="test", cate="Chair", category=None, **kwargs):
        print(kwargs)
        self.__dict__.update(locals())
        super().__init__()
        if self.category is not None:
            self.cate = cate = self.category
        self.data_dir = f"/studio/nnrecon/datasets/PartNet/eval_dset/{cate}/"
        self.length = len(glob.glob(self.data_dir+"*.npy"))//2
        print("dataset length", self.length)
    def __getitem__(self, index):
        Xct = np.load(self.data_dir+f"{index}_Xct.npy")
        Xbd = np.load(self.data_dir+f"{index}_Xbd.npy")
        return {"Xct":Xct, "Xbd":Xbd}
    def __len__(self):
        return self.length

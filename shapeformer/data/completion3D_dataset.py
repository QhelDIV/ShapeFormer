
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

class Completion3DDataset(Dataset):
    def __init__(self, dataset_path='datasets/completion3D/shapenet16384/', data_list=None, split='train', scale=1., **kwargs):
        super().__init__()
        self.__dict__.update(locals())
        self.split = split
        self.dpath = dpath = dataset_path
        if self.data_list is None:
            self.data_list = np.loadtxt(f"{dataset_path}/{split}.list", dtype=str)
        self.length = len(self.data_list)
        print("Dataset size: ", self.length)

    def __len__(self):
        return self.length
    def __getitem__(self, index):
        index = index % self.length
        if self.split!="test":
            complete_path = os.path.join(self.dataset_path, str(self.split), "gt",      self.data_list[index]+".h5")
            with h5py.File(complete_path, "r") as f:
                Xbd = f["data"][:] # 2048 or 16384
        else:
            Xbd = None

        partial_path  = os.path.join(self.dataset_path, str(self.split), "partial", self.data_list[index]+".h5")
        with h5py.File(partial_path,  "r")  as f:
            Xct = f["data"][:] # 2048

        #choice = np.random.choice(Xbd.shape[0], self.boundary_N, replace=True)
        #Xbd = Xbd[choice]
    
        item = dict(Xct  = Xct)
        if Xbd is not None:
            item["Xbd"] = Xbd
        for key in item:
            item[key] = item[key] * self.scale
            item[key][:,[0,1,2]] = item[key][:,[2,1,0]]
            item[key] = torch.from_numpy(item[key]).float()
        return item
    


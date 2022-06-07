
import numpy as np
from torch.utils.data import DataLoader, Dataset
import glob
import os
from pathlib import Path

"""
for cmp_hprscan
[793, 637, 604, 594, 496]
[41,179,188]

for 

[[3, 4, 16, 18, 20, 22], [3, 12, 14, 15, 17], [62, 80, 82,100,0,10,20,30,40,50]]
"""

class PaperDataset(Dataset):
    def __init__(self, split="test", load_keys=["Xbd", "Xct"], subsample=True, boundary_N=32768, context_N=16384, evalseed=314, data_dir="/studio/datasets/shapeformer/cmp_hprscan/test", **kwargs):
        super().__init__()
        self.__dict__.update(locals())
        assert self.split!="train", "This dataset only aims for test"
        self.shape_paths = glob.glob( os.path.join(data_dir,"*/") )
        print(self.shape_paths)

    def __len__(self):
        return len(self.shape_paths)
    def __getitem__(self, ind):
        shape_path = self.shape_paths[ind]
        ditem = {}
        for key in self.load_keys:
            ditem[key] = np.load(shape_path+f"/{key}.npy")
        if self.subsample==True:
            if "Xbd" in ditem:
                ditem["Xbd"] = ditem["Xbd"][ np.random.choice(ditem["Xbd"].shape[0], self.boundary_N) ]
            if "Xct" in ditem:
                ditem["Xct"] = ditem["Xct"][ np.random.choice(ditem["Xct"].shape[0], self.context_N) ]
        return ditem
class npyDataset(Dataset):    
    def __init__(self, split="test", dlist="/studio/datasets/shapeformer/cmp_hprscan/cmp_tests/test.lst", 
                    subsample=True, boundary_N=32768, context_N=16384, **kwargs):
        super().__init__()
        self.__dict__.update(locals())
        assert self.split!="train", "This dataset only aims for test"
        slist = np.loadtxt(dlist, dtype=str)
        self.shape_paths = slist
        print(self.shape_paths)

    def __len__(self):
        return len(self.shape_paths)
    def __getitem__(self, ind):
        shape_path = self.shape_paths[ind]
        data = np.load(shape_path, allow_pickle=True)
        loaded = data if "npz" in shape_path else data.item()
        ditem = dict(loaded)
        if self.subsample==True:
            if "Xbd" in ditem:
                ditem["Xbd"] = ditem["Xbd"][ np.random.choice(ditem["Xbd"].shape[0], self.boundary_N) ]
            if "Xct" in ditem:
                ditem["Xct"] = ditem["Xct"][ np.random.choice(ditem["Xct"].shape[0], self.context_N) ]
        for key in ditem:
            if type(ditem[key]) is np.ndarray:
                ditem[key] = ditem[key].astype(np.float32)
        return ditem
class ListDataset(Dataset):
    def __init__(self, ditem_list, split="test", load_keys=["Xbd", "Xct"], subsample=True, boundary_N=32768, context_N=16384, evalseed=314, data_dir="/studio/datasets/shapeformer/cmp_hprscan/test", **kwargs):
        super().__init__()
        self.__dict__.update(locals())

        self.ditem_paths = np.loadtxt(ditem_list)

    def __len__(self):
        return len(self.ditem_paths)
    def __getitem__(self, ind):
        ditem_path = self.ditem_paths[ind]
        ditem = {}
        for key in self.load_keys:
            ditem[key] = np.load(ditem_path+f"/{key}.npy")
        if self.subsample==True:
            if "Xbd" in ditem:
                ditem["Xbd"] = ditem["Xbd"][ np.random.choice(ditem["Xbd"].shape[0], self.boundary_N) ]
            if "Xct" in ditem:
                ditem["Xct"] = ditem["Xct"][ np.random.choice(ditem["Xct"].shape[0], self.context_N) ]
        return ditem

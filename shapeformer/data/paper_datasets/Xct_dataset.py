
import numpy as np
from torch.utils.data import DataLoader, Dataset
import glob
import os
from pathlib import Path
import open3d as o3d


class XctDataset(Dataset):
    def __init__(self, Xct_list, split="test", **kwargs):
        super().__init__()
        self.__dict__.update(locals())
        self.Xcts = np.loadtxt(Xct_list, dtype=str)

    def __len__(self):
        return len(self.Xcts)

    def __getitem__(self, ind):
        Xct = o3d.io.read_point_cloud(self.Xcts[ind])
        Xct = np.array(Xct.points).astype(np.float32)
        return {"Xct":  Xct}  # /np.abs(Xct).max()*.9  }

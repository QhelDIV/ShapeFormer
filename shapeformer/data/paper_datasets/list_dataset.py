
import numpy as np
from torch.utils.data import DataLoader, Dataset
import glob
import os
from pathlib import Path

"""
select point cloud data from a list
"""


class ListDataset(Dataset):
    def __init__(self, ditem_list, list_dir, split="test", load_keys=["Xbd", "Xct"], subsample=True, boundary_N=32768, context_N=16384, evalseed=314, **kwargs):
        super().__init__()
        self.__dict__.update(locals())

        self.ditem_names = np.loadtxt(ditem_list)
        self.list_dir = list_dir

    def __len__(self):
        return len(self.ditem_names)

    def __getitem__(self, ind):
        ditem_name = self.ditem_names[ind]
        ditem_path = os.path.join(self.list_dir, ditem_name)
        ditem = {}
        for key in self.load_keys:
            ditem[key] = np.load(ditem_path+f"/{key}.npy")
        if self.subsample == True:
            if "Xbd" in ditem:
                ditem["Xbd"] = ditem["Xbd"][np.random.choice(
                    ditem["Xbd"].shape[0], self.boundary_N)]
            if "Xct" in ditem:
                ditem["Xct"] = ditem["Xct"][np.random.choice(
                    ditem["Xct"].shape[0], self.context_N)]
        return ditem

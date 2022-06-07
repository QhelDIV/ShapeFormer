
import numpy as np
from torch.utils.data import DataLoader, Dataset
import glob
import os
from pathlib import Path
from xgutils import *
class SimpleDataset(Dataset):
    def __init__(self, split="test", **kwargs):
        super().__init__()
        self.__dict__.update(locals())
    def __len__(self):
        return 10
    def __getitem__(self, ind):
        Xct = Xbd = geoutil.sampleMesh(geoutil.cube["vert"], geoutil.cube["face"], sampleN=32768) #/ 6.
        print(np.abs(Xbd).max())
        return {"Xct":Xct, "Xbd":Xbd}

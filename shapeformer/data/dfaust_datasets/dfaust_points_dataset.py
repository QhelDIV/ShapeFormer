
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


class DFAUSTDataset(Dataset):
    def __init__(self, dataset_path='datasets/DFAUST/data/', data_list=None, split='train',
                 boundary_N=2048, target_N=8192, version=0, resolution=64, apply_random_shift=False,
                 partial_opt={"class": "shapeformer.data.ar_datasets.partial.BallSelector",
                              "kwargs": dict(radius=.4, context_N=512)}):
        super().__init__()
        self.__dict__.update(locals())
        self.split = split
        self.dpath = dpath = dataset_path
        if self.data_list is None:
            if split == "train":
                self.data_list = np.loadtxt(
                    "datasets/DFAUST/train.lst", dtype=str)
            if split == "test" or split == "val":
                self.data_list = np.loadtxt(
                    "datasets/DFAUST/val.lst", dtype=str)
        self.all_objs = []
        for data_name in self.data_list:
            self.all_objs.extend(glob.glob(dpath + data_name + "/*.obj"))
        self.length = len(self.all_objs)
        print("Dataset size: ", self.length)
        self.partial_selector = sysutil.instantiate_from_opt(self.partial_opt)
        self.Xtg = nputil.makeGrid(
            [-1, -1, -1.], [1., 1, 1], [resolution, ]*3, indexing="ij")

    def __len__(self):
        return self.length

    def apply_transform(self, Xbd):
        if self.apply_random_shift == True:
            shift = (np.random.rand(3)*.1)-.05  # increase generalizability
            Xbd -= shift[None, ...]
        return Xbd

    def __getitem__(self, index):
        index = index % self.length
        #vert, face = igl.read_triangle_mesh(self.all_objs[index])
        #Xbd = geoutil.sampleMesh( vert, face, self.boundary_N)

        suffix = f"_v{self.version}_res{self.resolution}.npy"
        npy_dpath = ".".join(self.all_objs[index].split(".")[:-1]) + suffix
        npy_data = np.load(npy_dpath, allow_pickle=True).item()

        Xbd = npy_data["Xbd"]
        if np.abs(Xbd).max() > 1.:
            print("Warning, data exceeds bbox 1.", index, np.abs(Xbd).max())
        Xct = np.float32(self.get_partial(Xbd))
        choice = np.random.choice(Xbd.shape[0], self.boundary_N, replace=True)
        Xbd = Xbd[choice]

        item = dict(Xct=Xct,
                    Xbd=Xbd,
                    )
        item = ptutil.nps2ths(item, device="cpu")
        return item

    def get_partial(self, Xbd, Xtg=None, Ytg=None):
        Xct = self.partial_selector(Xbd)
        return Xct

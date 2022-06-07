from scipy.spatial.transform import Rotation as R
import torch
import sklearn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from xgutils import *
import scipy
import h5py
import time
import glob
import os
import igl
#
""" 
total_shapes: 43783
data structure: 
    cate_id/shape_id/...
                     model.binvox: (32,32,32) voxelized shape
                     points.npz : 
                        points: (100000,3)
                        occupancies: (12500,) need np.unpackbits to get (100000,)
                        scale: ()
                        loc: (3,)
                     pointcloud.npz: 
                        points: (100000,3)
                        normals: (100000,3)
                        scale: ()
                        loc: (3,)
                     img_choy2016: ...
data stats: cate_id, name, total_num(train:test:val -> 7:2:1)
    04256520 sofa 3173
    02691156 airplane 4045
    03636649 lamp 2318
    04401088 telephone 1052
    04530566 vessel 1939
    03691459 loudspeaker 1618
    03001627 chair 6778
    02933112 cabinet 1572
    04379243 table 8509
    03211117 display 1095
    02958343 car 7496
    02828884 bench 1816
    04090263 rifle 2372
"""
id2name = {"04256520": "sofa",    "02691156": "airplane",    "03636649": "lamp",    "04401088": "telephone",    "04530566": "vessel",    "03691459": "loudspeaker",
           "03001627": "chair",    "02933112": "cabinet",    "04379243": "table",    "03211117": "display",    "02958343": "car",    "02828884": "bench",    "04090263": "rifle", }
name2id = {"sofa": "04256520",    "airplane": "02691156",    "lamp": "03636649",    "telephone": "04401088",    "vessel": "04530566",    "loudspeaker": "03691459",
           "chair": "03001627",    "cabinet": "02933112",    "table": "04379243",    "display": "03211117",    "car": "02958343",    "bench": "02828884",    "rifle": "04090263", }


class OccnetDataset(Dataset):
    def __init__(self, data_dir='/studio/datasets/occnet/', split='test', boundary_N=32768, target_N=-1,
                 seed=314, scale=1.9, cate=["lamp"],
                 partial_opt={"class": "shapeformer.data.ar_datasets.partial.VirtualScanSelector",
                                "kwargs": dict(radius=10, context_N=16384)}):
        self.__dict__.update(locals())
        if type(self.cate) is str and self.cate == "all":
            self.cate = list(name2id.keys())
        self.shapes = []
        for ci in self.cate:
            lst = np.loadtxt(
                f"{data_dir}/{name2id[ci]}/{split}.lst", dtype=str)
            for shapeid in lst:
                self.shapes.append(f"{data_dir}/{name2id[ci]}/{shapeid}")
        self.length = len(self.shapes)
        self.partial_selector = sysutil.instantiate_from_opt(self.partial_opt)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        Xbd_path = os.path.join(self.shapes[index], "pointcloud.npz")
        XYtg_path = os.path.join(self.shapes[index], "points.npz")
        Xbd_loaded = np.load(Xbd_path)
        XYtg_loaded = np.load(XYtg_path)
        Xbd = Xbd_loaded["points"]
        Xtg = XYtg_loaded["points"].astype(np.float32)
        Ytg = np.unpackbits(XYtg_loaded["occupancies"]).astype(
            np.float32)[..., None]

        with nputil.temp_seed(self.seed):
            Xct = self.partial_selector(Xbd)
            if self.boundary_N != -1:
                Xbd = Xbd[np.random.choice(
                    Xbd.shape[0], self.boundary_N, replace=True)]
            if self.target_N != -1:
                tg_choice = np.random.choice(
                    Xtg.shape[0], self.target_N, replace=True)
                Xtg = Xtg[tg_choice]
                Ytg = Ytg[tg_choice]

        Xbd = Xbd.astype(np.float32)
        Xct = Xct.astype(np.float32)

        Xbd = Xbd[:, [2, 1, 0]]*self.scale
        Xtg = Xtg[:, [2, 1, 0]]*self.scale
        Xct = Xct[:, [2, 1, 0]]*self.scale
        return {"Xbd": Xbd, "Xct": Xct, "Xtg": Xtg, "Ytg": Ytg}

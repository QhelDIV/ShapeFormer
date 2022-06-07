from shapeformer.data.ar_datasets.partial import HalfSpaceSelector
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

from scipy.spatial.transform import Rotation as R


class FamousDataset(Dataset):
    def __init__(self, data_dir='/studio/datasets/famous_aligned/', split='test', boundary_N=8192, target_N=-1, evalseed=314, scale=0.8,
                 partial_opt={"class": "shapeformer.data.ar_datasets.partial.BallSelector",
                                "kwargs": dict(radius=.4, context_N=512)}):
        self.__dict__.update(locals())
        self.objs = glob.glob(os.path.join(data_dir, "*.ply"))
        self.length = len(self.objs)
        self.partial_selector = sysutil.instantiate_from_opt(self.partial_opt)

    def __len__(self):
        return self.length*1000

    def __getitem__(self, index):
        index = index % self.length
        objname = self.objs[index].split("/")[-1]
        vert, face = igl.read_triangle_mesh(self.objs[index])
        vert, face = vert.astype(np.float32), face.astype(int)
        with nputil.temp_seed(self.evalseed):
            Xbd = geoutil.sampleMesh(vert, face, sampleN=self.boundary_N)
            #rot = R.from_euler('zyx', [45, 0, 0], degrees=True)
            #Xbd = rot.apply(Xbd)
            Xct = self.get_partial(Xbd)

        Xbd = Xbd.astype(np.float32)
        Xct = Xct.astype(np.float32)
        return {"Xbd": Xbd, "Xct": Xct, "vert": vert, "face": face, "name": objname}

    def get_partial(self, Xbd, Xtg=None, Ytg=None):
        Xct = self.partial_selector(Xbd)
        return Xct


class FamousAlignedDataset(Dataset):
    def __init__(self, data_dir='/studio/datasets/famous_aligned/', split='test', duplicate_size=3, boundary_N=8192, target_N=-1, evalseed=314, scale=0.8,
                 partial_opt={"class": "shapeformer.data.ar_datasets.partial.BallSelector",
                                "kwargs": dict(radius=.4, context_N=512)}, **kwargs):
        self.__dict__.update(locals())
        self.objs = glob.glob(os.path.join(data_dir, "*.obj"))
        self.length = len(self.objs)
        self.partial_selector = sysutil.instantiate_from_opt(self.partial_opt)

    def __len__(self):
        return self.length * self.duplicate_size

    def __getitem__(self, index):
        ori_ind = index
        index = index % self.length
        objname = self.objs[index].split("/")[-1]
        vert, face = igl.read_triangle_mesh(self.objs[index])
        vert, face = vert.astype(np.float32), face.astype(int)
        vert[:, [0, 1, 2]] = vert[:, [0, 2, 1]]

        with nputil.temp_seed(self.evalseed+ori_ind):
            Xbd = geoutil.sampleMesh(vert, face, sampleN=self.boundary_N)
            #rot = R.from_euler('zyx', [45, 0, 0], degrees=True)
            #Xbd = rot.apply(Xbd)
            Xbd -= (Xbd.max(axis=0) + Xbd.min(axis=0)) / 2
            Xbd /= np.abs(Xbd).max()
            print(index)
            if index == 0:
                scale = .9
            else:
                scale = .6
            Xbd *= scale

            cviews = np.array(
                [[0, 0, 5.], [0, 0, -5], [0, -5, 0], [0, -5, -2]])
            Xct = geoutil.hidden_point_removal(
                Xbd, cviews[(ori_ind//2) % cviews.shape[0]] + np.random.rand(3))

        Xbd = Xbd.astype(np.float32)
        Xct = Xct.astype(np.float32)
        return {"Xbd": Xbd, "Xct": Xct}

    def get_partial(self, Xbd, Xtg=None, Ytg=None):
        Xct = self.partial_selector(Xbd)
        return Xct


class ListFamousAlignedDataset(Dataset):
    def __init__(self, shape_list='/studio/nnrecon/research/CVPR22/famous/teapot.txt', split='test', boundary_N=8192, evalseed=314, **kwargs):
        # path, scale, camPos_x, camPos_y, camPos_z
        self.__dict__.update(locals())
        self.shape_cfgs = np.loadtxt(shape_list, dtype=str)
        self.length = len(self.shape_cfgs)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        shape_path = self.shape_cfgs[index][0]
        scale = float(self.shape_cfgs[index][1])
        camPos = float(self.shape_cfgs[index][2]), float(
            self.shape_cfgs[index][3]), float(self.shape_cfgs[index][4])
        camPos = np.array(camPos)

        vert, face = igl.read_triangle_mesh(shape_path)
        vert, face = vert.astype(np.float32), face.astype(int)
        vert[:, [0, 1, 2]] = vert[:, [0, 2, 1]]

        Xbd = geoutil.sampleMesh(vert, face, sampleN=self.boundary_N)
        Xbd -= (Xbd.max(axis=0) + Xbd.min(axis=0)) / 2
        Xbd /= np.abs(Xbd).max()
        Xbd *= scale
        Xct = geoutil.hidden_point_removal(Xbd, camPos*5)

        Xbd = Xbd.astype(np.float32)
        Xct = Xct.astype(np.float32)
        return {"Xbd": Xbd, "Xct": Xct}

    def get_partial(self, Xbd, Xtg=None, Ytg=None):
        Xct = self.partial_selector(Xbd)
        return Xct


class FamousTestDataset(FamousAlignedDataset):
    def __init__(self, shape_id=21, linspace_args=(0.1, .9, 20), direction=[0, 0, 1], **kwargs):
        self.__dict__.update(locals())
        super().__init__(**kwargs)
        self.portions = np.linspace(*linspace_args)
        self.selector = HalfSpaceSelector(
            0., portion_on="distance", context_N=16384, plane_normal=direction)

    def __len__(self):
        return self.linspace_args[2]

    def __getitem__(self, index):
        self.selector.portion = self.portions[index]
        data = super().__getitem__(self.shape_id)
        Xct = self.selector(data["Xbd"])
        return {"Xbd": data["Xbd"], "Xct": Xct}

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
    def __init__(self, data_dir='/studio/liqiang/buildings/', split='train', boundary_N=8192, target_N=-1,
                 partial_opt={"class": "shapeformer.data.ar_datasets.partial.BallSelector",
                                "kwargs": dict(radius=.4, context_N=512)}, **kwargs):
        self.__dict__.update(locals())
        #self.objs = glob.glob( os.path.join(data_dir,"*.ply") )
        self.objs = glob.glob(os.path.join(data_dir, "*.obj"))
        self.length = len(self.objs)
        self.partial_selector = sysutil.instantiate_from_opt(self.partial_opt)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        objname = self.objs[index].split("/")[-1]
        #vert, face = igl.read_triangle_mesh(self.objs[index])
        vert, _, _, face, _, _ = igl.read_obj(self.objs[index])
        vert, face = vert.astype(np.float32), face.astype(int)
        Xbd = geoutil.sampleMesh(vert, face, sampleN=self.boundary_N)
        Xbd -= (Xbd.max(axis=0) + Xbd.min(axis=0)) / 2
        Xbd /= np.abs(Xbd).max()
        Xbd = Xbd*0.7

        #rot = R.from_euler('zyx', [45, 0, 0], degrees=True)
        #Xbd = rot.apply(Xbd)
        Xct = self.get_partial(Xbd)

        Xbd = Xbd.astype(np.float32)
        Xct = Xct.astype(np.float32)
        return {"Xbd": Xbd, "Xct": Xct, "vert": vert, "face": face, "name": objname}
        # return {"Xbd":Xbd, "Xct":Xbd, "vert":vert, "face":face, "name":objname}

    def get_partial(self, Xbd, Xtg=None, Ytg=None):
        Xct = self.partial_selector(Xbd)
        return Xct

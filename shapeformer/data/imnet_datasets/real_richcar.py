
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
import sys

from shapeformer.data.ar_datasets.imnet_datasets import Imnet2Dataset, Imnet2LowResDataset


class Richcar_dataset(Imnet2LowResDataset):
    def __init__(self, split="test", samples_per_cate=100, shape_ind="01833", context_N=8192, camR=10, evalseed=314, **kwargs):
        kwargs["partial_opt"] = {"class": "shapeformer.data.ar_datasets.partial.CamVirtualScanSelector",
                                 "kwargs": dict(radius=camR, context_N=context_N)}
        partial_opt = {"class": "shapeformer.data.ar_datasets.partial.CamVirtualScanSelector",
                       "kwargs": dict(radius=camR, context_N=context_N)}
        kwargs["split"] = split
        super().__init__(**kwargs)
        self.__dict__.update(locals())
        assert self.split != "train", "This dataset only aims for test"
        cate_list = [0, 1, 3, 4, 6, 9, 10, 12]
        # 0 plane, 1 bench, 2 car, 3 chair, 4 lamp, 5 sofa, 6 table 7 vessel
        # 0plane, 1bench, 2cabinet, 3car, 4chair, 5tv, 6lamp, 7cab, 8gun, 9sofa, 10table, 11?, 12vessel
        choices = np.zeros((len(cate_list), samples_per_cate), dtype=np.int)
        with h5py.File(self.dpath, 'r') as f:
            with nputil.temp_seed(evalseed):
                for i, cate in enumerate(cate_list):
                    cate_cand = np.array(f[f"cate_{cate}"])
                    #print(cate, cate_cand)
                    choices[i] = np.random.choice(
                        cate_cand.shape[0], samples_per_cate)
                    choices[i] = cate_cand[choices[i]]

        self.shapeids = choices.transpose(1, 0).reshape(-1)

        self.redwood_path = '/studio/liqiang/redwood/bak/'
        self.pts_files = glob.glob(os.path.join(
            self.redwood_path, f"{shape_ind}.pts"))
        self.partial_selector = sysutil.instantiate_from_opt(partial_opt)
        self.camera_poses = [np.array([0.5, 0.5, -1.8]),
                             np.array([0, .25, -2]),
                             np.array([0, 0, 2]),
                             np.array([0, 2, 0]),
                             np.array([1, 1, 1]),
                             np.array([2, 2, 2]),
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             geoutil.sample_sphere(1)[0]*2,
                             ]

        points = np.loadtxt(self.pts_files[0])[:, :3].astype(np.float32)
        # points = np.loadtxt(self.pts_files[0])[:,:3].astype(np.float32)
        points -= (points.max(axis=0) + points.min(axis=0)) / 2
        points /= np.abs(points).max()
        points = points*0.85
        self.points = points

    def __len__(self):
        # return len(self.shapeids)
        return len(self.camera_poses)

    def __getitem__(self, ind):
        #shape_i = self.shapeids[ind]
        # with nputil.temp_seed( (self.evalseed+ind)%123456 ):
        #    ditem = super().__getitem__(shape_i)
        # return ditem
        # return {"Xbd":points, "Xct":points}
        points = self.points
        return {"Xbd": points, "Xct": self.partial_selector(points, camera_pos=self.camera_poses[ind])}


class HPRScanHD_dataset(Imnet2Dataset):
    def __init__(self, split="test", samples_per_cate=100, context_N=8192, camR=10, evalseed=314, **kwargs):
        kwargs["partial_opt"] = {"class": "shapeformer.data.ar_datasets.partial.VirtualScanSelector",
                                 "kwargs": dict(radius=camR, context_N=context_N)}
        kwargs["split"] = split
        super().__init__(**kwargs)
        self.__dict__.update(locals())
        assert self.split != "train", "This dataset only aims for test"
        cate_list = [0, 1, 3, 4, 6, 9, 10, 12]
        # 0 plane, 1 bench, 2 car, 3 chair, 4 lamp, 5 sofa, 6 table 7 vessel
        # 0plane, 1bench, 2cabinet, 3car, 4chair, 5tv, 6lamp, 7cab, 8gun, 9sofa, 10table, 11?, 12vessel
        choices = np.zeros((len(cate_list), samples_per_cate), dtype=np.int)
        with h5py.File(self.dpath, 'r') as f:
            with nputil.temp_seed(evalseed):
                for i, cate in enumerate(cate_list):
                    cate_cand = np.array(f[f"cate_{cate}"])
                    #print(cate, cate_cand)
                    choices[i] = np.random.choice(
                        cate_cand.shape[0], samples_per_cate)
                    choices[i] = cate_cand[choices[i]]

        self.shapeids = choices.transpose(1, 0).reshape(-1)

    def __len__(self):
        return len(self.shapeids)

    def __getitem__(self, ind):
        shape_i = self.shapeids[ind]
        with nputil.temp_seed((self.evalseed+ind) % 123456):
            ditem = super().__getitem__(shape_i)
        return ditem

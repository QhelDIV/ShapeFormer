
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


class Redwood(Imnet2LowResDataset):
    def __init__(self, split="test", samples_per_cate=100, context_N=8192, camR=10, evalseed=314, **kwargs):
        # kwargs["partial_opt"] = {   "class"  : "shapeformer.data.ar_datasets.partial.fixedVirtualScanSelector",
        kwargs["partial_opt"] = {"class": "shapeformer.data.ar_datasets.partial.VirtualScanSelector",
                                 "kwargs": dict(radius=camR, context_N=context_N)}
        kwargs["split"] = split
        super().__init__(**kwargs)
        self.__dict__.update(locals())
        # self.partial_selector = sysutil.instantiate_from_opt(self.partial_opt)
        # assert self.split!="train", "This dataset only aims for test"
        # cate_list = [0, 1, 3, 4, 6, 9, 10, 12]
        # # 0 plane, 1 bench, 2 car, 3 chair, 4 lamp, 5 sofa, 6 table 7 vessel
        # # 0plane, 1bench, 2cabinet, 3car, 4chair, 5tv, 6lamp, 7cab, 8gun, 9sofa, 10table, 11?, 12vessel
        # choices = np.zeros((len(cate_list), samples_per_cate), dtype=np.int)
        # with h5py.File(self.dpath,'r') as f:
        #     with nputil.temp_seed(evalseed):
        #         for i, cate in enumerate(cate_list):
        #             cate_cand  = np.array(f[f"cate_{cate}"])
        #             #print(cate, cate_cand)
        #             choices[i] = np.random.choice(cate_cand.shape[0], samples_per_cate)
        #             choices[i] = cate_cand[choices[i]]

        # self.shapeids = choices.transpose(1,0).reshape(-1)
        self.redwood_path = '/studio/liqiang/redwood'
        self.pts_files = glob.glob(os.path.join(self.redwood_path, "*.pts"))
        self.partial_selector = sysutil.instantiate_from_opt(partial_opt)

    def __len__(self):
        # return len(self.shapeids)
        return len(self.pts_files)

    def __getitem__(self, ind):
        # shape_i = self.shapeids[ind]
        # with nputil.temp_seed( (self.evalseed+ind)%123456 ):
        #     ditem = super().__getitem__(shape_i)
        # return ditem
        points = np.loadtxt(self.pts_files[ind])[:, :3].astype(np.float32)
        points[:, 0] -= np.mean(points[:, 0])
        points[:, 1] -= np.mean(points[:, 1])
        points[:, 2] -= np.mean(points[:, 2])
        points /= points.max()
        points = points*0.7
        # import pdb; pdb.set_trace()
        return {"Xbd": points, "Xct": self.get_partial(points)}


class Redwood2(Imnet2LowResDataset):
    def __init__(self, split="test", samples_per_cate=100, context_N=8192, camR=10, evalseed=314, **kwargs):
        # kwargs["partial_opt"] = {   "class"  : "shapeformer.data.ar_datasets.partial.fixedVirtualScanSelector",
        kwargs["partial_opt"] = {"class": "shapeformer.data.ar_datasets.partial.VirtualScanSelector",
                                 "kwargs": dict(radius=camR, context_N=context_N)}
        kwargs["split"] = split
        super().__init__(**kwargs)
        self.__dict__.update(locals())
        # self.partial_selector = sysutil.instantiate_from_opt(self.partial_opt)
        # assert self.split!="train", "This dataset only aims for test"
        # cate_list = [0, 1, 3, 4, 6, 9, 10, 12]
        # # 0 plane, 1 bench, 2 car, 3 chair, 4 lamp, 5 sofa, 6 table 7 vessel
        # # 0plane, 1bench, 2cabinet, 3car, 4chair, 5tv, 6lamp, 7cab, 8gun, 9sofa, 10table, 11?, 12vessel
        # choices = np.zeros((len(cate_list), samples_per_cate), dtype=np.int)
        # with h5py.File(self.dpath,'r') as f:
        #     with nputil.temp_seed(evalseed):
        #         for i, cate in enumerate(cate_list):
        #             cate_cand  = np.array(f[f"cate_{cate}"])
        #             #print(cate, cate_cand)
        #             choices[i] = np.random.choice(cate_cand.shape[0], samples_per_cate)
        #             choices[i] = cate_cand[choices[i]]

        # self.shapeids = choices.transpose(1,0).reshape(-1)
        self.redwood_path = '/studio/liqiang/redwood'
        self.pts_files = glob.glob(os.path.join(self.redwood_path, "*.pts"))
        self.partial_selector = sysutil.instantiate_from_opt(partial_opt)

    def __len__(self):
        # return len(self.shapeids)
        return len(self.pts_files)

    def __getitem__(self, ind):
        # shape_i = self.shapeids[ind]
        # with nputil.temp_seed( (self.evalseed+ind)%123456 ):
        #     ditem = super().__getitem__(shape_i)
        # return ditem
        points = np.loadtxt(self.pts_files[ind])[:, :3].astype(np.float32)
        points -= (points.max(axis=0) + points.min(axis=0)) / 2
        points /= np.abs(points).max()
        points = points*0.9
        # import pdb; pdb.set_trace()
        return {"Xbd": points, "Xct": self.get_partial(points)}


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

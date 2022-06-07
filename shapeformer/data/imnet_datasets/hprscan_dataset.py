
import contextlib
import torch
import sklearn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from xgutils import *
import scipy
import h5py
import time

from shapeformer.data.ar_datasets.imnet_datasets import Imnet2Dataset, Imnet2LowResDataset


class HPRScan_dataset(Imnet2LowResDataset):
    def __init__(self, split="test", samples_per_cate=100, context_N=8192, camR=10, evalseed=314, manual_cameras={},
                 classname="shapeformer.data.ar_datasets.partial.VirtualScanSelector", **kwargs):
        kwargs["partial_opt"] = {"class": classname,
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

    def convert_index(self, index):
        index = self.shapeids[index]
        return index

    def __getitem__(self, ind):
        shape_i = self.shapeids[ind]  # self.convert_index(ind)
        with nputil.temp_seed((self.evalseed+ind) % 123456):
            ditem = super().__getitem__(shape_i)
        return ditem


class OrthoHPRScan_dataset(HPRScan_dataset):
    def __init__(self, classname="shapeformer.data.ar_datasets.partial.OrthoVirtualScanSelector", **kwargs):
        super().__init__(classname=classname, **kwargs)


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


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

# math utils


class AMScan_dataset(Dataset):
    def __init__(self, split="test", cate_list="all", class_label=False, gen_xct=True, dpath="/studio/datasets/IMNet2_64/",
                 am_range=[.9, 1.], samples_per_cate=100, context_N=16384, boundary_N=32768, evalseed=314, random_choice=False,
                 fixed_camPos=None, random_views=False, Xbd_as_Xct=False, Ytg=False,
                 **kwargs):
        super().__init__()
        self.__dict__.update(locals())
        if split == "train":
            self.random_views = True
        #assert self.split!="train", "This dataset only aims for test"
        self.h5path = f"{self.dpath}/{self.split}.hdf5"
        #cate_list = [0, 1, 3, 4, 6, 9, 10, 12]
        # 0 plane, 1 bench, 2 car, 3 chair, 4 lamp, 5 sofa, 6 table 7 vessel
        # 0plane, 1bench, 2cabinet, 3car, 4chair, 5tv, 6lamp, 7cab, 8gun, 9sofa, 10table, 11?, 12vessel
        if type(cate_list) is str and cate_list == "all":
            cate_list = np.arange(13)
        if samples_per_cate == -1:
            cates = []
            labels = []
            with h5py.File(self.h5path, 'r') as f:
                for i, cate in enumerate(cate_list):
                    cates.append(np.array(f[f"cate_{cate}"]))
                    labels.append(np.zeros(cates[-1].shape[0]) + i)
            self.shapeids = np.concatenate(cates)
            self.labels = np.concatenate(labels)
        else:
            choices = np.zeros(
                (len(cate_list), samples_per_cate), dtype=np.int)
            with h5py.File(self.h5path, 'r') as f:
                with temp_seed(evalseed):
                    for i, cate in enumerate(cate_list):
                        cate_cand = np.array(f[f"cate_{cate}"])
                        print(cate, cate_cand.shape[0])
                        if self.random_choice == True:
                            choices[i] = np.random.choice(
                                cate_cand.shape[0], samples_per_cate)
                        else:
                            choices[i] = np.arange(self.samples_per_cate)
                            choices[i][choices[i] >= cate_cand.shape[0]
                                       ] = cate_cand.shape[0] - 1
                        choices[i] = cate_cand[choices[i]]
            self.shapeids = choices.transpose(1, 0).reshape(-1)

        cviews = geoutil.fibonacci_sphere(samples=64)
        ortho_views = np.array(
            [[1., 0, 0], [-1, 0, 0], [0, 1., 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        self.cviews = np.concatenate([cviews, ortho_views])*10

    def __len__(self):
        return len(self.shapeids)

    def __getitem__(self, ind):
        ditem = {}
        shape_i = self.shapeids[ind]

        seed = self.evalseed+shape_i
        if self.random_views == True:
            seed = seed + np.random.randint(1000)
        seed = seed % 123456

        if self.fixed_camPos is not None:
            cam_pos = np.array(self.fixed_camPos)
        else:
            if self.split == "train":
                vec = np.random.randn(1, 3)
                vec /= np.linalg.norm(vec, axis=1)[..., None]
                cam_pos = vec[0]
            else:
                viewranking = np.loadtxt(
                    self.dpath+f"/viewranks/{self.split}/{shape_i}.txt").astype(int)  # [::-1]
                with temp_seed(seed):
                    lb = np.round(viewranking.shape[0] * self.am_range[0])
                    ub = np.round(viewranking.shape[0] * self.am_range[1])
                    view = np.random.randint(lb, ub)
                    cam_pos = self.cviews[viewranking[view]]
        with temp_seed(seed):
            with h5py.File(self.h5path) as f:
                Xbd = np.array(f["Xbd"][shape_i])
                #Xbd = Xbd[np.random.choice(Xbd.shape[0], self.boundary_N)]
                if self.gen_xct == True and self.Xbd_as_Xct == False:
                    Xct = geoutil.hidden_point_removal(Xbd, cam_pos)
                else:
                    Xct = Xbd
                if self.Ytg == True:
                    ditem["Ytg"] = np.unpackbits(np.array(f["Ytg"][shape_i]))
        with temp_seed(seed):
            Xct = Xct[np.random.choice(Xct.shape[0], self.context_N)]
            Xbd = Xbd[np.random.choice(Xbd.shape[0], self.boundary_N)]
        ditem.update({"Xct": Xct, "Xbd": Xbd})
        if self.class_label == True:
            ditem["label"] = self.labels[ind].astype(int)
        return ditem

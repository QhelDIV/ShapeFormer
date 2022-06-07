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
import os
import traceback

default_dset_opt = {
    "class": "shapeformer.data.ar_datasets.imnet_datasets.Imnet2Dataset",
    "kwargs": {
        "split": "train",
        "dataset": "IMNet2_packed",
        "cate": "all",  # 0plane, 1bench, 2cabinet, 3car, 4tv, 5chair, 6lamp, 7cab, 8gun, 9sofa, 10table, 11?, 12vessel
        "boundary_N": 32768,
        "weighted_sampling": False,
        "grid_dim": 256,
        "target_N": -1,
        "duplicate_size": 1,
        "partial_opt": {
                "class": "shapeformer.data.ar_datasets.partial.VirtualScanSelector",
                "kwargs": {"context_N": 16384}
        }
    }
}


class implicit_sampler():
    def __init__(self, dataroot="/studio/datasets/IMNet2/", dset_opt=default_dset_opt, sample_N=64**3, version="v1"):
        self.__dict__.update(locals())
        self.dset = sysutil.instantiate_from_opt(opt=self.dset_opt)
        self.out_dir = os.path.join(
            dataroot, f"sampled_dset_{version}", self.dset_opt["kwargs"]["split"])
        sysutil.mkdirs(self.out_dir)

    def __len__(self):
        return 10
        return len(self.dset)

    def __call__(self, index):
        try:
            shape_pn = os.path.join(self.out_dir, str(index))
            sysutil.mkdirs(shape_pn)
            #out_path   = shape_pn + f"_{version}.npy"
            #shape_vocab, vocab_idx, Xbd = generate_dataitem(shape_path, selem_size=selem_size)
            #data = dict(shape_vocab=shape_vocab, vocab_idx=vocab_idx, Xbd=Xbd)
            #loaded = np.load( out_path, allow_pickle=True).item()
            ditem = self.dset[index]
            print(ditem["Ytg"].sum() / 256**3)
            vert, face = geoutil.array2mesh(
                ditem["Ytg"], dim=3, coords=ditem["Xtg"], thresh=.5, if_decimate=False)

            Xbd, Xtg, Ytg = geoutil.SDF_sampling(
                vert, face, sample_N=self.sample_N, near_std=0.015, far_std=0.2)
            Ytg = nputil.sigmoid(-Ytg)
            np.save(os.path.join(shape_pn, "Xbd.npy"), Xbd)
            np.save(os.path.join(shape_pn, "Xtg.npy"), Xtg)
            np.save(os.path.join(shape_pn, "Ytg.npy"), Ytg)
            #Xbd, Ytg = generate_dataitem(shape_path, selem_size=selem_size)
            #data = dict(Xbd=Xbd, Ytg=Ytg)
            #np.save( out_path, data)
        except Exception as e:
            traceback.print_exc()
            print(f'Error encountered during processing:{shape_path}', e)
            return 1
        return 0


def sampling_dset(dataroot, dset_opt, run_name="train"):
    sampler = implicit_sampler(dataroot=dataroot, dset_opt=dset_opt)
    #return_codes = sysutil.parallelMap(sampler, [np.arange(len(sampler))], zippedIn=False)
    #return_codes = sysutil.parallelMap(sampler, [np.arange(10)], zippedIn=False)
    sampler(100)
    #np.save(f"{dataroot}/{run_name}_sampling_failure_code.npy", return_codes)
    #print("Percentage of failure:", np.array(return_codes).sum()/len(sampler))
    #print("Return code:", return_codes)


def voxelize_imnet(dataroot="/studio/datasets/IMNet2/"):
    dset_opt = default_dset_opt
    dset_opt["kwargs"]["split"] = "train"
    sampling_dset(dataroot, dset_opt, run_name="train")
    print("train set sampled")
    dset_opt["kwargs"]["split"] = "test"
    sampling_dset(dataroot, dset_opt, run_name="test")
    print("test set sampled")

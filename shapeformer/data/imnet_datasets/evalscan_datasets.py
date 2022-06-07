
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from xgutils import *
import scipy
import h5py
import time
from .imnet_datasets import Imnet2Dataset
class EvalScanDataset(Dataset):
    def __init__(self, dataset='IMNet2_packed', cate="all", zoomfac=1, grid_dim=256, Xct_as_Xbd=False, \
                 split='test', boundary_N=8192, context_N=8192, target_N=-1, **kwargs):
        self.__dict__.update(locals())
        super().__init__()
        self.split    = split
        self.dpath    = f'datasets/{dataset}/{split}.hdf5'
        self.scan_choice  = np.load("/studio/datasets/IMNet2_eval/scan_choice.npy").astype(int)
        self.dset_indices = np.load("/studio/datasets/IMNet2_eval/dset_indices.npy").astype(int)
        self.scan_choice  = self.scan_choice.reshape(-1)
        self.dset_indices = self.dset_indices.reshape(-1)
        self.scan_dir = f"/studio/datasets/IMNet2_scan_256/{split}/"
        self.length = len(self.dset_indices)
        print("dset length", self.length)
        self.Xbds = nputil.H5Var(self.dpath, "Xbd")
        self.shape_vocabs = nputil.H5Var(self.dpath, "shape_vocab")
        self.vocab_idxs    = nputil.H5Var(self.dpath, "vocab_idx")
        self.all_Xtg = nputil.makeGrid([-1,-1,-1.],[1.,1,1],[grid_dim,]*3, indexing="ij")

    def __len__(self):
        return self.length
    def __getitem__(self, dind, all_target=False):
        dind = dind % self.length
        index = self.dset_indices[dind]
        print("index", index)
        with h5py.File(self.dpath,'r') as f:
            Xbd = np.array(f["Xbd"][index]).astype(np.float32)
            #Xct = np.float32(self.get_partial(Xbd))
            Xct = np.load(self.scan_dir+f"{index}_s{self.scan_choice[dind]}.npy").astype(np.float32)
            choice = np.random.choice(Xbd.shape[0], self.boundary_N, replace=True)
            Xbd = Xbd[choice]
            choice = np.random.choice(Xct.shape[0], self.context_N, replace=True)
            Xct = Xct[choice]
        shape_vocab, vocab_idx = self.shape_vocabs[index], self.vocab_idxs[index]
        tgx, tgy = self.get_target(shape_vocab, vocab_idx, all_target=all_target)
        if self.Xct_as_Xbd == True:
            Xbd = Xct
        item = dict(Xct  = Xct,
                    Xbd  = Xbd,
                    Xtg  = tgx,
                    Ytg  = tgy,
                    )
        #item = ptutil.nps2ths(item)
        item = ptutil.ths2nps(item)
        return item
    
    def get_partial(self, Xbd, Xtg=None, Ytg=None):
        Xct = self.partial_selector(Xbd)
        return Xct
    def get_target(self, shape_vocab, vocab_idx, all_target=False):
        voxels = ptutil.decompress_voxels(shape_vocab, vocab_idx)
        x_dim, grid_dim = len(voxels.shape), voxels.shape[-1]
        if self.target_N==-1 or all_target==True:
            Xtg = self.all_Xtg
            Ytg = voxels.reshape(-1,1)
        else:
            if self.weighted_sampling ==True:
                rdind_uniform = torch.randint(0,grid_dim, (self.target_N//2, x_dim))
                flat = voxel.reshape(-1)
                inside_pos = np.where(flat)[0]
                outside_pos = np.where(flat)[0]
                rdc1 = np.random.choice(len(inside_pos),  self.target_N//2)
                rdc2 = np.random.choice(len(outside_pos), self.target_N//2)
                choice = np.concatenate([inside_pos[rdc1], outside_pos[rdc2]])
                inds = ptutil.unravel_index(torch.from_numpy(choice), shape=(256,256,256))
            else:
                inds = torch.randint(0,grid_dim, (self.target_N, x_dim))
            Xtg = ptutil.index2point(inds, grid_dim=grid_dim).numpy()
            Ytg = voxels[inds[:,0], inds[:,1], inds[:,2]][...,None]
            # allind = ptutil.unravel_index(torch.arange(grid_dim**3), (256,256,256))
            # Xtg = ptutil.index2point(allind, grid_dim=grid_dim).numpy()
            # Ytg = voxels[allind[:,0], allind[:,1], allind[:,2]][...,None]
        return Xtg, Ytg
    @classmethod
    def unittest(cls, **kwargs):
        train_dset = cls(boundary_N=102400, target_N=8192, **kwargs)
        ts = []
        ts.append(time.time())
        ditems=[train_dset[i] for i in range(32)]
        ts.append(time.time())
        print("training dataloading time", ts[-1]-ts[-2])

        dset = cls(boundary_N=102400, **kwargs)
        ts.append(time.time())
        ditems=[dset[i] for i in range(1)]
        ts.append(time.time())
        print("dataloading time", ts[-1]-ts[-2])
        ditem = ditems[0]
        voxelized = ptutil.point2voxel(torch.from_numpy(ditem["Xbd"])[None,...], grid_dim=64).reshape(-1).numpy()
        Xtg64 = nputil.makeGrid([-1,-1,-1],[1,1,1],[64,64,64])
        img = npfvis.plot_3d_sample(Xct=ditem["Xct"], Yct=None, Xtg=Xtg64, Ytg=voxelized, pred_y=ditem["Ytg"], \
                show_images=["GT"])[1][0]
        visutil.showImg(img)

        imgs = [npfvis.plot_3d_sample(Xct=ditem["Xct"], Yct=None, Xtg=ditem["Xtg"], Ytg=ditem["Ytg"], pred_y=ditem["Ytg"], \
                show_images=["GT"])[1][0] for ditem in ditems]
        ts.append(time.time())
        print("plot time", ts[-1]-ts[-2])
        print(imgs[0].shape)
        grid= visutil.imageGrid(imgs, zoomfac=1)
        visutil.showImg(grid)
        return dset, ditems, imgs, grid

def make_test_dataset():
    dset = Imnet2Dataset(split="test")
    cate_list = [0, 1, 3, 4, 6, 9, 10, 12]
    # 0 plane, 1 bench, 2 car, 3 chair, 4 lamp, 5 sofa, 6 table 7 vessel
    # 0plane, 1bench, 2cabinet, 3car, 4chair, 5tv, 6lamp, 7cab, 8gun, 9sofa, 10table, 11?, 12vessel
    num = 10
    scan_num = 8
    dset_indices = np.zeros((len(cate_list), num), dtype=int)
    with h5py.File(dset.dpath,"r") as f:
        for i, catei in enumerate(cate_list):
            chs = np.array(f[f"cate_{catei}"])
            print(chs.shape)
            dset_indices[i] = chs[ np.random.choice(chs.shape[0], num) ]
    scan_choice = np.random.randint(0, scan_num, dset_indices.shape)
    np.save("/studio/datasets/IMNet2_eval/scan_choice.npy",  scan_choice)
    np.save("/studio/datasets/IMNet2_eval/dset_indices.npy", dset_indices)


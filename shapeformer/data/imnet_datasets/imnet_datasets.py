
import torch
import sklearn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from xgutils import *
import scipy
import h5py
import time


class Imnet2Dataset(Dataset):
    def __init__(self, dataset='IMNet2_packed', cate="all", zoomfac=1,
                 duplicate_size=1, split='train', boundary_N=2048, target_N=-1, grid_dim=256, weighted_sampling=False,
                 partial_opt={"class": "shapeformer.data.ar_datasets.partial.BallSelector",
                               "kwargs": dict(radius=.4, context_N=512)}):
        self.split = split
        self.grid_dim = grid_dim
        self.dpath = dpath = f'datasets/{dataset}/{split}.hdf5'
        self.weighted_sampling = weighted_sampling
        with h5py.File(dpath, 'r') as f:
            total_length = f['Xbd'].shape[0]
            all_ind = np.arange(total_length)
            if type(cate) is str:
                if cate == "all":
                    self.subset = all_ind
                else:
                    self.subset = np.array(f[f"cate_{cate}"])
            elif type(cate) is list:
                self.subset = np.concatenate(
                    [np.array(f[f"cate_{cat}"]) for cat in cate])
                print("subset num", len(self.subset))
        self.length = len(self.subset)
        if split != "train":
            self.duplicate_size = 1
        else:
            self.duplicate_size = duplicate_size
        self.split = split
        self.boundary_N, self.target_N = boundary_N, target_N
        self.partial_opt = partial_opt
        self.partial_selector = sysutil.instantiate_from_opt(self.partial_opt)

        self.Xbds = nputil.H5Var(self.dpath, "Xbd")
        self.shape_vocabs = nputil.H5Var(self.dpath, "shape_vocab")
        self.vocab_idxs = nputil.H5Var(self.dpath, "vocab_idx")

        self.all_Xtg = nputil.makeGrid(
            [-1, -1, -1.], [1., 1, 1], [grid_dim, ]*3, indexing="ij")

    def __len__(self):
        return self.length * self.duplicate_size

    def __getitem__(self, index, all_target=False):
        index = index % self.length
        index = self.subset[index]
        with h5py.File(self.dpath, 'r') as f:
            Xbd = self.Xbds[index]
            Xct = np.float32(self.get_partial(Xbd))
            choice = np.random.choice(
                Xbd.shape[0], self.boundary_N, replace=True)
            Xbd = Xbd[choice]
            shape_vocab, vocab_idx = self.shape_vocabs[index], self.vocab_idxs[index]
            tgx, tgy = self.get_target(
                shape_vocab, vocab_idx, all_target=all_target)

            item = dict(Xct=Xct,
                        Xbd=Xbd,
                        Xtg=tgx,
                        Ytg=tgy,
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
        if self.target_N == -1 or all_target == True:
            Xtg = self.all_Xtg
            Ytg = voxels.reshape(-1, 1)
        else:
            if self.weighted_sampling == True:
                rdind_uniform = torch.randint(
                    0, grid_dim, (self.target_N//2, x_dim))
                flat = voxel.reshape(-1)
                inside_pos = np.where(flat)[0]
                outside_pos = np.where(flat)[0]
                rdc1 = np.random.choice(len(inside_pos),  self.target_N//2)
                rdc2 = np.random.choice(len(outside_pos), self.target_N//2)
                choice = np.concatenate([inside_pos[rdc1], outside_pos[rdc2]])
                inds = ptutil.unravel_index(
                    torch.from_numpy(choice), shape=(256, 256, 256))
            else:
                inds = torch.randint(0, grid_dim, (self.target_N, x_dim))
            Xtg = ptutil.index2point(inds, grid_dim=grid_dim).numpy()
            Ytg = voxels[inds[:, 0], inds[:, 1], inds[:, 2]][..., None]
            # allind = ptutil.unravel_index(torch.arange(grid_dim**3), (256,256,256))
            # Xtg = ptutil.index2point(allind, grid_dim=grid_dim).numpy()
            # Ytg = voxels[allind[:,0], allind[:,1], allind[:,2]][...,None]
        return Xtg, Ytg

    @classmethod
    def unittest(cls, **kwargs):
        train_dset = cls(boundary_N=102400, target_N=8192, **kwargs)
        ts = []
        ts.append(time.time())
        ditems = [train_dset[i] for i in range(32)]
        ts.append(time.time())
        print("training dataloading time", ts[-1]-ts[-2])

        dset = cls(boundary_N=102400, **kwargs)
        ts.append(time.time())
        ditems = [dset[i] for i in range(1)]
        ts.append(time.time())
        print("dataloading time", ts[-1]-ts[-2])
        ditem = ditems[0]
        voxelized = ptutil.point2voxel(torch.from_numpy(ditem["Xbd"])[
                                       None, ...], grid_dim=64).reshape(-1).numpy()
        Xtg64 = nputil.makeGrid([-1, -1, -1], [1, 1, 1], [64, 64, 64])
        img = npfvis.plot_3d_sample(Xct=ditem["Xct"], Yct=None, Xtg=Xtg64, Ytg=voxelized, pred_y=ditem["Ytg"],
                                    show_images=["GT"])[1][0]
        visutil.showImg(img)

        imgs = [npfvis.plot_3d_sample(Xct=ditem["Xct"], Yct=None, Xtg=ditem["Xtg"], Ytg=ditem["Ytg"], pred_y=ditem["Ytg"],
                show_images=["GT"])[1][0] for ditem in ditems]
        ts.append(time.time())
        print("plot time", ts[-1]-ts[-2])
        print(imgs[0].shape)
        grid = visutil.imageGrid(imgs, zoomfac=1)
        visutil.showImg(grid)
        return dset, ditems, imgs, grid

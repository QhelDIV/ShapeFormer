
import scipy.io as sio
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
                 duplicate_size=1, split='train', boundary_N=2048, target_N=-1, grid_dim=256, weighted_sampling=False, Xbd_as_Xct=False, Xct_as_Xbd=False,
                 partial_opt={"class": "shapeformer.data.partial.BallSelector",
                               "kwargs": dict(radius=.4, context_N=512)}):
        self.__dict__.update(locals())
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
        o_ind = index
        index = self.subset[index]
        with h5py.File(self.dpath, 'r') as f:
            Xbd = self.Xbds[index]
            Xct = np.float32(self.get_partial(Xbd, o_ind))
            choice = np.random.choice(
                Xbd.shape[0], self.boundary_N, replace=True)
            Xbd = Xbd[choice]
            shape_vocab, vocab_idx = self.shape_vocabs[index], self.vocab_idxs[index]
            Xtg, Ytg = self.get_target(
                shape_vocab, vocab_idx, all_target=all_target)

            if self.Xct_as_Xbd == True:
                Xbd = Xct
            print("Xtg", Xtg.shape, Ytg.shape)
            item = dict(Xct=Xct,
                        Xbd=Xbd,
                        Xtg=Xtg,
                        Ytg=Ytg,
                        )
            #item = ptutil.nps2ths(item)
            item = ptutil.ths2nps(item)
        return item

    def get_partial(self, Xbd, Xtg=None, Ytg=None, index=None):
        Xct = self.partial_selector(Xbd, index=index)
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


class Imnet2LowResDataset(Dataset):
    def __init__(self, dataset='IMNet2_64', cate="all", zoomfac=1,
                 duplicate_size=1, split='train', boundary_N=2048, target_N=-1, grid_dim=64, weighted_sampling=False, Xbd_as_Xct=False, Xct_as_Xbd=False,
                 partial_opt={"class": "shapeformer.data.partial.BallSelector",
                               "kwargs": dict(radius=.4, context_N=512)}):
        self.__dict__.update(locals())
        self.split = split
        self.dpath = dpath = f'datasets/{dataset}/{split}.hdf5'
        self.weighted_sampling = weighted_sampling
        self.grid_dim = grid_dim
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
        self.Ytgs = nputil.H5Var(self.dpath, "Ytg")  # packed

        self.all_Xtg = nputil.makeGrid(
            [-1, -1, -1.], [1., 1, 1], [grid_dim, ]*3, indexing="ij")

    def __len__(self):
        return self.length * self.duplicate_size

    def __getitem__(self, index, all_target=False):
        index = index % self.length
        o_ind = index
        index = self.subset[index]
        with h5py.File(self.dpath, 'r') as f:
            Xbd = self.Xbds[index]
            Xct = np.float32(self.get_partial(Xbd, index=o_ind))
            choice = np.random.choice(
                Xbd.shape[0], self.boundary_N, replace=True)
            Xbd = Xbd[choice]
            x_dim = Xbd.shape[-1]

            Ytg = np.unpackbits(self.Ytgs[index], axis=-1)[..., None]
            Xtg = self.all_Xtg
            if self.weighted_sampling == True:
                target_N = self.target_N if self.target_N != - \
                    1 else Xtg.shape[0]
                Xtg, Ytg = balanced_sampling2(
                    Xbd, Xtg, Ytg, target_N=target_N, x_dim=x_dim, grid_dim=self.grid_dim)
            else:
                if self.target_N != -1 and all_target == False:
                    choice = np.random.choice(
                        Xtg.shape[0], self.target_N, replace=True)
                    Xtg = Xtg[choice]
                    Ytg = Ytg[choice]
            if self.Xct_as_Xbd == True:
                Xbd = Xct
            item = dict(Xct=Xct.astype(np.float32),
                        Xbd=Xbd.astype(np.float32),
                        Xtg=Xtg.astype(np.float32),
                        Ytg=Ytg.astype(np.float32),
                        )
            #item = ptutil.nps2ths(item,)
        return item

    def get_partial(self, Xbd, Xtg=None, Ytg=None, index=None):
        if self.Xbd_as_Xct == True:
            return Xbd
        Xct = self.partial_selector(Xbd, index=index)
        return Xct

    @classmethod
    def unittest(cls, grid_dim=32, **kwargs):
        train_dset = cls(grid_dim=grid_dim, boundary_N=102400,
                         target_N=8192, **kwargs)
        ts = []
        ts.append(time.time())
        ditems = [train_dset[i] for i in range(32)]
        ts.append(time.time())
        print("training dataloading time", ts[-1]-ts[-2])

        boundary_N = 4096
        dset = cls(boundary_N=boundary_N, **kwargs)
        ts.append(time.time())
        ditems = [dset[i] for i in range(1)]
        ts.append(time.time())
        print("dataloading time", ts[-1]-ts[-2])
        ditem = ditems[0]
        print(f"point res={256} {boundary_N}, grid_dim={64} point2voxel")
        voxelized = ptutil.point2voxel(torch.from_numpy(ditem["Xbd"])[
                                       None, ...], grid_dim=64).reshape(-1).numpy()
        Xtg64 = nputil.makeGrid([-1, -1, -1], [1, 1, 1], [64, 64, 64])
        img = npfvis.plot_3d_sample(Xct=ditem["Xct"], Yct=None, Xtg=Xtg64, Ytg=voxelized, pred_y=ditem["Ytg"],
                                    show_images=["GT"])[1][0]
        visutil.showImg(img)

        print(f"ground truth grid_dim={grid_dim}")
        imgs = [npfvis.plot_3d_sample(Xct=ditem["Xct"], Yct=None, Xtg=ditem["Xtg"], Ytg=ditem["Ytg"], pred_y=ditem["Ytg"],
                show_images=["GT"])[1][0] for ditem in ditems]
        ts.append(time.time())
        print("plot time", ts[-1]-ts[-2])
        print(imgs[0].shape)
        grid = visutil.imageGrid(imgs, zoomfac=1)
        visutil.showImg(grid)
        return dset, ditems, imgs, grid

# def vis_imnet(dset, vis_dir):
#     vis_camera = dict(camPos=np.array([2,2,2]), camLookat=np.array([0.,0.,0.]),\
#         camUp=np.array([0,1,0]),camHeight=2,resolution=(256,256), samples=16)
#     for i in range(len(dset)):
#         item = dset[i]
#         item["Xct"]
#         item["Xbd"]
#         item["Xtg"]
#         item["Ytg"]


def balanced_sampling(Xbd, Xtg, Ytg, target_N=4096, x_dim=3, grid_dim=32):
    rdind_uniform = torch.randint(0, grid_dim, (target_N//2, x_dim))
    inside_pos = np.where(Ytg)[0]
    outside_pos = np.where(1-Ytg)[0]
    rdc_xbd = np.random.choice(Xbd.shape[0], target_N//2, replace=True)
    sub_Xbd = Xbd[rdc_xbd]

    rdc1 = np.random.choice(len(inside_pos),  target_N//4, replace=True)
    rdc2 = np.random.choice(len(outside_pos), target_N//4, replace=True)
    choice = np.concatenate([rdc_xbd, inside_pos[rdc1], outside_pos[rdc2]])
    #inds = ptutil.unravel_index(torch.from_numpy(choice), shape=(256,256,256))
    sub_Xtg = np.concatenate([Xtg[choice], sub_Xbd])
    sub_Ytg = np.concatenate([Ytg[choice], np.zeros((sub_Xbd.shape[0], 1))+.5])
    return sub_Xtg, sub_Ytg


def balanced_sampling2(Xbd, Xtg, Ytg, target_N=4096, x_dim=3, grid_dim=32, random_scale=.1):
    rdind_uniform = torch.randint(0, grid_dim, (target_N//2, x_dim))
    rdc_xbd = np.random.choice(Xbd.shape[0], target_N//2, replace=True)
    sub_Xbd = Xbd[rdc_xbd] + np.random.randn(len(rdc_xbd), x_dim)*random_scale
    sub_Xbd_ind = ptutil.point2index(torch.from_numpy(
        sub_Xbd), grid_dim=grid_dim, ravel=True).numpy()
    sub_Xbd_Y = Ytg[sub_Xbd_ind]

    rdc1 = np.random.choice(Xtg.shape[0],  target_N//2, replace=True)
    choice = np.concatenate([rdc_xbd, rdc1])
    #inds = ptutil.unravel_index(torch.from_numpy(choice), shape=(256,256,256))
    sub_Xtg = Xtg[choice]
    sub_Ytg = Ytg[choice]
    #sub_Xtg = np.concatenate([Xtg[choice], sub_Xbd])
    #sub_Ytg = np.concatenate([Ytg[choice], sub_Xbd_Y])
    return sub_Xtg, sub_Ytg


def generate_dataitem(shape_path):
    loaded = sio.loadmat(shape_path)  # , allow_pickle=True)
    shape_vocab, vocab_idx = loaded["b"].reshape(
        loaded["b"].shape[0], -1), (loaded["bi"]-1).astype(int).reshape(-1)
    folded = ptutil.decompress_voxels(shape_vocab, vocab_idx, unpackbits=False)
    folded = geoutil.shapenetv1_to_shapenetv2(folded)
    folded = geoutil.shapenetv2_to_cart(folded)
    shape_vocab, vocab_idx = ptutil.compress_voxels(folded, packbits=True)

    v256, f256 = geoutil.array2mesh(folded.reshape(-1), dim=3, bbox=np.array(
        [[-1, -1, -1], [1, 1, 1.]])*1., thresh=.5, if_decimate=False, cart_coord=True)
    Xbd = geoutil.sampleMesh(v256, f256, 65536)

    return shape_vocab, vocab_idx, Xbd


def make_imnet_dataset():
    imnet_datapath = "datasets/IM-NET"
    hspnet_datapath = "datasets/hsp_shapenet"
    imnet_path = os.path.join(imnet_datapath, "IMSVR/data")
    hspnet_path = os.path.join(hspnet_datapath, "modelBlockedVoxels256")
    train_shapeh5 = imnet_path + "/all_vox256_img_train.hdf5"
    test_shapeh5 = imnet_path + "/all_vox256_img_test.hdf5"
    lines = open(imnet_path+"/all_vox256_img_train.txt", "r").readlines()
    train_shape_names = [line.strip() for line in lines]
    lines = open(imnet_path+"/all_vox256_img_test.txt", "r").readlines()
    test_shape_names = [line.strip() for line in lines]
    target_dir = "datasets/IMNet2_packed/"
    sysutil.mkdirs(target_dir)

    typelist = [train_shape_name.split("/")[0]
                for train_shape_name in train_shape_names]
    unique_types = np.unique(typelist)
    type_dict = dict([(typ, i) for i, typ in enumerate(unique_types)])

    cates = [[] for typ in type_dict]
    for si in range(len(train_shape_names)):
        shape_name = train_shape_names[si]
        type_ind = type_dict[shape_name.split("/")[0]]
        cates[type_ind].append(si)
    shape_paths = [hspnet_path+"/"+shape_name +
                   ".mat" for shape_name in train_shape_names]
    shape_vocabs, vocab_idxs, Xbds = sysutil.parallelMap(
        generate_dataitem, [shape_paths], zippedIn=False)
    dataDict = {"shape_vocab": np.array(shape_vocabs, dtype="O"), "vocab_idx": np.array(
        vocab_idxs), "Xbd": np.array(Xbds)}
    for ci in range(len(cates)):
        dataDict[f"cate_{ci}"] = np.array(cates[ci])
    nputil.writeh5(target_dir+"train.hdf5", dataDict)

    cates = [[] for typ in type_dict]
    for si in range(len(test_shape_names)):
        shape_name = test_shape_names[si]
        type_ind = type_dict[shape_name.split("/")[0]]
        cates[type_ind].append(si)
    shape_paths = [hspnet_path+"/"+shape_name +
                   ".mat" for shape_name in test_shape_names]
    shape_vocabs, vocab_idxs, Xbds = sysutil.parallelMap(
        generate_dataitem, [shape_paths], zippedIn=False)
    dataDict = {"shape_vocab": np.array(shape_vocabs, dtype="O"), "vocab_idx": np.array(
        vocab_idxs), "Xbd": np.array(Xbds)}
    for ci in range(len(cates)):
        dataDict[f"cate_{ci}"] = np.array(cates[ci])
    nputil.writeh5(target_dir+"test.hdf5", dataDict)


def IMNet2_h5_unittest():
    dataDict = nputil.readh5("datasets/IMNet2/test.h5")
    print(dataDict.keys())
    print([dataDict["shape_vocab"][i].shape for i in range(10)])
    a, b, c = dataDict["shape_vocab"][0], dataDict["vocab_idx"][0], dataDict["Xbd"][0]
    #a,b,c = generate_dataitem("datasets/hsp_shapenet/modelBlockedVoxels256/04530566/cc3957e0605cd684bb48c7922d71f3d0.mat")
    unfold = ptutil.decompress_voxels(a, b)
    dflt_camera = dict(camPos=np.array([2, 2, 2]), camLookat=np.array([0., 0., 0.]),
                       camUp=np.array([0, 1, 0]), camHeight=2.414, resolution=(512, 512), samples=256)
    vert, face = geoutil.array2mesh(unfold.reshape(-1), dim=3, bbox=np.array(
        [[-1, -1, -1], [1, 1, 1.]])*1., thresh=.5, if_decimate=False, cart_coord=True)
    print(f'vert shape: {vert.shape}, face shape: {face.shape}')
    gtmesh = fresnelvis.renderMeshCloud(
        mesh={'vert': vert, 'face': face}, cloud=c, cloudR=0.001, axes=True, **dflt_camera)
    visutil.showImg(gtmesh)
# IMNet2_h5_unittest()

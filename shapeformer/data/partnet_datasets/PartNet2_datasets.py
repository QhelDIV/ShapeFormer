"""
Contrary to PartNet_datasets.py
These datasets shrink the partnet pc by .9 in order to make the mesh fall into [-1,1] bounding box.
Though there are still some shapes violate this principle, most of them satisfy the rule.
Also, in these datasets Ytg (ground truth voxel value) are available
"""
import torch
import sklearn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from xgutils import *
import scipy
import h5py
import time
import os
import random
import trimesh
import csv
import json
#from util.pc_utils import rotate_point_cloud_by_axis_angle, sample_point_cloud_by_n

#SPLIT_DIR = "/studio/Multimodal-Shape-Completion/data/partnet_train_val_test_split"
PARTNET_DIR = "/studio/datasets/PartNet/"
SPLIT_DIR = "/studio/datasets/PartNet/partnet_train_val_test_split"
#PC_MERGED_LABEL_DIR = "/studio/Multimodal-Shape-Completion/data/partnet_pc_label"
PC_MERGED_LABEL_DIR = "/studio/datasets/PartNet/partnet_pc_label"

def rotate_point_cloud(points, transformation_mat):

    new_points = np.dot(transformation_mat, points.T).T

    return new_points
def rotate_point_cloud_by_axis_angle(points, axis, angle_deg):
    """ align 3depn shapes to shapenet coordinates"""
    # angle = math.radians(angle_deg)
    # rot_m = pymesh.Quaternion.fromAxisAngle(axis, angle)
    # rot_m = rot_m.to_matrix()
    rot_m = np.array([[ 2.22044605e-16,  0.00000000e+00,  1.00000000e+00],
                      [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
                      [-1.00000000e+00,  0.00000000e+00,  2.22044605e-16]])

    new_points = rotate_point_cloud(points, rot_m)

    return new_points
def downsample_point_cloud(points, n_pts):
    """downsample points by random choice

    :param points: (n, 3)
    :param n_pts: int
    :return:
    """
    p_idx = random.choices(list(range(points.shape[0])), k=n_pts)
    return points[p_idx]
def upsample_point_cloud(points, n_pts):
    """upsample points by random choice

    :param points: (n, 3)
    :param n_pts: int, > n
    :return:
    """
    p_idx = random.choices(list(range(points.shape[0])), k=n_pts - points.shape[0])
    dup_points = points[p_idx]
    points = np.concatenate([points, dup_points], axis=0)
    return points
def sample_point_cloud_by_n(points, n_pts):
    """resample point cloud to given number of points"""
    if n_pts > points.shape[0]:
        return upsample_point_cloud(points, n_pts)
    elif n_pts < points.shape[0]:
        return downsample_point_cloud(points, n_pts)
    else:
        return points


def collect_data_id(split_dir, classname, phase):
    filename = os.path.join(split_dir, "{}.{}.json".format(classname, phase))
    if not os.path.exists(filename):
        raise ValueError("Invalid filepath: {}".format(filename))

    all_ids = []
    with open(filename, 'r') as fp:
        info = json.load(fp)
    for item in info:
        all_ids.append(item["anno_id"])

    return all_ids
def PartNet2cart(x):
    return x
class PartNetDataset(Dataset):
    """ Modified from MSC """
    def __init__(self, split, data_root, category, n_pts, n_tg=32768, balanced_tg=False, **kwargs):
        super().__init__()
        if split == "validation":
            split = "val"
        self.phase = self.split = phase = split
        self.aug = phase == split
        self.n_tg = n_tg 
        self.balanced_tg = balanced_tg

        self.data_root = data_root

        shape_names = collect_data_id(SPLIT_DIR, category, phase)
        self.shape_names = []
        for name in shape_names:
            path = os.path.join(PC_MERGED_LABEL_DIR, name)
            if os.path.exists(path):
                self.shape_names.append(name)

        self.n_pts = n_pts
        self.raw_n_pts = self.n_pts // 2

        self.rng = random.Random(1234)
        self.all_Xtg = torch.from_numpy(nputil.makeGrid([-1,-1,-1.],[1.,1,1],[64,]*3, indexing="ij")).float()

    @staticmethod
    def load_point_cloud(path):
        pc = trimesh.load(path)
        pc = pc.vertices * .9 # scale to [-1,1]^3 box
        return pc

    @staticmethod
    def read_point_cloud_part_label(path):
        with open(path, 'r') as fp:
            labels = fp.readlines()
        labels = np.array([int(x) for x in labels])
        return labels

    def random_rm_parts(self, raw_pc, part_labels):
        part_ids = sorted(np.unique(part_labels).tolist())
        if self.phase == "train":
            random.shuffle(part_ids)
            n_part_keep = random.randint(1, max(1, len(part_ids) - 1))
        else:
            self.rng.shuffle(part_ids)
            n_part_keep = self.rng.randint(1, max(1, len(part_ids) - 1))
        part_ids_keep = part_ids[:n_part_keep]
        point_idx = []
        for i in part_ids_keep:
            point_idx.extend(np.where(part_labels == i)[0].tolist())
        raw_pc = raw_pc[point_idx]
        return raw_pc, n_part_keep

    def __getitem__(self, index, **kwargs):
        shape_name = self.shape_names[index]
        real_ply_path = os.path.join(self.data_root, shape_name, 'point_sample/ply-10000.ply')
        real_pc = self.load_point_cloud(real_ply_path)

        raw_label_path = os.path.join(PC_MERGED_LABEL_DIR, shape_name, 'label-merge-level1-10000.txt')
        part_labels = self.read_point_cloud_part_label(raw_label_path)
        # new labels start from 0 to k, whereas part_labels may be like [2,10,9,2,9]
        part_ids, part_new_labels = np.unique(part_labels, return_inverse=True)
        raw_pc, n_part_keep = self.random_rm_parts(real_pc, part_labels)
        raw_pc = sample_point_cloud_by_n(raw_pc, self.raw_n_pts)

        real_pc = np.load(os.path.join(self.data_root, shape_name, "sampled_32768.npy"))
        
        real_pc = torch.tensor(real_pc).float()
        raw_pc  = torch.tensor(raw_pc).float()

        real_pc = PartNet2cart(real_pc)
        raw_pc  = PartNet2cart(raw_pc)

        voxel_path = os.path.join(self.data_root, shape_name, "morph_voxel_64_1.npy")
        Ytg = np.load(voxel_path).reshape(-1)
        Ytg = torch.from_numpy(Ytg).float()[...,None]
        Xtg = self.all_Xtg.clone()
        if self.split=="train" and self.balanced_tg==True:
            zero_list = np.where(Ytg.numpy()==0)[0]
            one_list  = np.where(Ytg.numpy()==1)[0]
            num_zeros = self.n_tg*7//8
            num_ones  = self.n_tg - num_zeros
            zchoice   = zero_list[np.random.choice(len(zero_list), num_zeros)]
            ochoice   = one_list[np.random.choice(len(one_list),  num_ones)]
            choice    = np.concatenate([zchoice, ochoice])
            Xtg       = Xtg[choice]
            Ytg       = Ytg[choice]
        return {"Xct": raw_pc, "Xbd": real_pc, "Xtg": Xtg, "Ytg": Ytg, 
                "n_part_keep": n_part_keep, "part_new_labels": part_new_labels}

    def __len__(self):
        return len(self.shape_names)


# 0,2,10,11,
# 24,32,42,113,
# 209,845,1156,1085,
# 1192,1066,927,857,
# 107,850,935,1178,

class EvalDatasetPartNet(PartNetDataset):
    def __init__(self, subset=list(range(20)), **kwargs):
        print(kwargs)
        self.subset = subset
        super().__init__(**kwargs)
    def __getitem__(self, index, **kwargs):
        if self.split=="test":
            index = self.subset[index]
        return super().__getitem__(index, **kwargs)
    def __len__(self):
        if self.split=="test":
            return len(self.subset)
        else:
            return super().__len__()
import glob
import igl
def voxelize_partnet_shape(shape_dir, grid_dim=64, shrink_percentage=.9, selem_size=1):
    try:
        objs = glob.glob( os.path.join(shape_dir, "objs/*") )
        meshes = []
        for obj in objs:
            vert, face=igl.read_triangle_mesh(obj)
            meshes.append({"vert":vert, "face":face})
        merged = geoutil.mergeMeshes(meshes)
        merged["vert"] *= shrink_percentage
        sampled_32768 = geoutil.sampleMesh(merged["vert"], merged["face"], sampleN=32768)
        #voxel, coords = geoutil.morph_voxelization(merged["vert"], merged["face"], grid_dim=grid_dim, selem_size=selem_size)
        #Ytg = voxel.reshape(-1)
        #np.save( os.path.join(shape_dir, f"morph_voxel_{grid_dim}_{selem_size}.npy"), Ytg)
        np.save( os.path.join(shape_dir, f"sampled_32768.npy"), sampled_32768)
    except Exception as e:
        print(f'Error encountered during voxelization:{shape_dir}', e)
        return 1
    return 0
def voxelize_partnet(data_root="/studio/datasets/PartNet/data_v0/", grid_dim=64):
    shape_dirs = glob.glob( os.path.join(data_root, "*") )
    #for shape_dir in sysutil.progbar(shape_dirs):
    #    print(shape_dir)
    #    voxelize_partnet_shape(shape_dir)
    
    return_codes = sysutil.parallelMap(voxelize_partnet_shape, [shape_dirs], zippedIn=False)
    np.save("/studio/datasets/PartNet/voxelization_failure_code.npy",return_codes)
    print("Percentage of failure:", np.array(return_codes).sum()/len(shape_dirs))
    print("Return code:", return_codes)

# TODO, need to modify these two classes
class ProgPartNet(Dataset):
    def __init__(self, split, data_root, category, n_pts, **kwargs):
        super().__init__()
        if split == "validation":
            split = "val"
        self.phase = self.split = phase = split
        self.aug = phase == split


        self.data_root = data_root

        shape_names = collect_data_id(SPLIT_DIR, category, phase)
        self.shape_names = []
        for name in shape_names:
            path = os.path.join(PC_MERGED_LABEL_DIR, name)
            if os.path.exists(path):
                self.shape_names.append(name)
        self.precompute_accumulated()

        self.n_pts = n_pts
        self.raw_n_pts = self.n_pts# // 2

        self.rng = random.Random(1234)
        self.all_Xtg = torch.from_numpy(nputil.makeGrid([-1,-1,-1.],[1.,1,1],[64,]*3, indexing="ij")).float()

    @staticmethod
    def load_point_cloud(path):
        pc = trimesh.load(path)
        pc = pc.vertices * .9 # scale to [-1,1]^3 box
        return pc

    @staticmethod
    def read_point_cloud_part_label(path):
        with open(path, 'r') as fp:
            labels = fp.readlines()
        labels = np.array([int(x) for x in labels])
        return labels

    def get_part_labels(self,shape_name):
        raw_label_path = os.path.join(PC_MERGED_LABEL_DIR, shape_name, 'label-merge-level1-10000.txt')
        part_labels = self.read_point_cloud_part_label(raw_label_path)
        # new labels start from 0 to k, whereas part_labels may be like [2,10,9,2,9]
        part_ids, part_inv_labels = np.unique(part_labels, return_inverse=True)
        return part_ids, part_labels, part_inv_labels
    def get_parts(self, index, pc, labels):
        """ labels should take value from 0 to k-1
        """
        shape_ind = np.searchsorted(self.accumulated_parts, index) - 1
        part_num  = index - self.accumulated_parts[shape_ind]
        partial_pc = pc[labels<part_num]
        return partial_pc
    def precompute_accumulated(self):
        store_path = PARTNET_DIR+"accumulated_parts.npy"
        if os.path.exists(store_path):
            self.accumulated_parts = np.load(store_path)
            return
        self.accumulated_parts = np.zeros(len(self.shape_names)+1, dtype=int)
        for index in range(len(self.shape_names)):
            shape_name = self.shape_names[index]
            part_ids, part_labels, part_inv_labels = self.get_part_labels(shape_name)
            parts_num = len(part_ids)
            self.accumulated_parts[index+1] = parts_num
        self.accumulated_parts = np.cumsum(self.accumulated_parts)
        np.save(store_path, self.accumulated_parts)
        print(store_path, "Saved!")

    def __getitem__(self, index, **kwargs):
        shape_ind = np.searchsorted(self.accumulated_parts, index, side="right") - 1
        n_part_keep = part_num  = index - self.accumulated_parts[shape_ind]
        shape_name = self.shape_names[shape_ind]
        
        real_ply_path = os.path.join(self.data_root, shape_name, 'point_sample/ply-10000.ply')
        real_pc = self.load_point_cloud(real_ply_path)

        part_ids, part_labels, part_inv_labels = self.get_part_labels(shape_name)
        raw_pc = real_pc[part_inv_labels<=part_num]
        raw_pc = sample_point_cloud_by_n(raw_pc, self.raw_n_pts)

        
        real_pc = torch.tensor(real_pc, dtype=torch.float32)
        raw_pc  = torch.tensor(raw_pc, dtype=torch.float32)

        real_pc = PartNet2cart(real_pc)
        raw_pc  = PartNet2cart(raw_pc)

        voxel_path = os.path.join(self.data_root, shape_name, "morph_voxel_64_1.npy")
        Ytg = np.load(voxel_path).reshape(-1)
        Ytg = torch.from_numpy(Ytg).float()

        return {"Xct": raw_pc, "Xbd": real_pc, "Xtg": self.all_Xtg, "Ytg":Ytg, 
                "n_part_keep": n_part_keep, "part_inv_labels": part_inv_labels}

    def __len__(self):
        return self.accumulated_parts[-1]

class EvalDatasetProgPartNet(ProgPartNet):
    def __init__(self, subset=list(range(30)), **kwargs):
        self.subset = subset
        super().__init__(**kwargs)
    def __getitem__(self, index, **kwargs):
        return super().__getitem__(index, **kwargs)
    def __len__(self):
        if self.split=="test":
            return self.accumulated_parts[len(self.subset)]
        else:
            return super().__len__()

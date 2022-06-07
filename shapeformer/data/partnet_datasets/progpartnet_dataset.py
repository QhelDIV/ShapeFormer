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

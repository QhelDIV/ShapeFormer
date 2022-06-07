'''


'''
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

from shapeformer.util import util
from shapeformer.data import utils as datautils
from shapeformer import vis
from shapeformer.data.basenp_dataset import BaseNPDataset


class Furn4000Dataset(BaseNPDataset):
    '''
        0~999: bench
        1000~1999: chair
        2000~2999: sofa
        3000~3999: table

        default visuals for all category [3567, 2954, 358, 3929]
        default visuals for all sofa [3567, 2954, 358, 3929]
    '''

    def default_opt(self):
        return {
            'datah5':       'furn4000.h5',
            'selectIds':    'sofa',
            'shots':        2048,
            'multiplier':   1,
            'vis_indices':  [],
            'points_target': False,
        }

    def __init__(self, opt, dataDict=None, split='train'):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        super().__init__(opt, dataDict, split)
        self.datapath = os.path.join(self.opt.datasets_dir, self.opt.datah5)
        with h5py.File(self.datapath, 'r') as f:
            self.split_train = np.array(f['split_train'])
            self.split_test = np.array(f['split_test'])
        self.all_x = util.H5Var(self.datapath, 'sdf_x')
        self.all_y = util.H5Var(self.datapath, 'sdf_y')
        self.points = util.H5Var(self.datapath, 'points')
        self.partial_params = util.H5Var(self.datapath, 'partial_params')
        self.select = self.split_train if split == 'train' else self.split_test
        self.multiplier = self.opt.multiplier if split == 'train' else 1

        zeros = np.zeros(4000, dtype=int)
        ifselect = np.zeros(4000, dtype=bool)
        if type(self.opt.selectIds) is str:
            filt = zeros
            if self.opt.selectIds == 'bench':
                filt[0:1000] = True
            if self.opt.selectIds == 'chair':
                filt[1000:2000] = True
            if self.opt.selectIds == 'sofa':
                filt[2000:3000] = True
            if self.opt.selectIds == 'table':
                filt[3000:4000] = True
            self.select = self.select[np.where(filt[self.select])[0]]
        else:
            self.select = np.array(self.opt.selectIds, dtype=int)
        if split == 'val' or split == 'test':
            # for i,ind in enumerate(self)
            self.all_points = self.points[self.select]
            self.points = dict([(ind, self.all_points[i])
                               for i, ind in enumerate(self.select)])
        self.length = self.select.shape[0]
        total_num = self.split_train.shape[0] + self.split_test.shape[0]
        self.parse_vis_index(length=total_num)
        #self.perm = np.arange(self.length)
        #print('Dataset initialized')

    def __getitem__(self, index):
        index = index % self.length
        index = self.select[index]
        context_x = self.points[index]
        #print(index, self.points[index])
        #print(index, self.pointss[index])
        context_y = np.zeros((context_x.shape[0], 1))
        if self.split != 'train':
            choice = partial_select(context_x, self.partial_params[index])
            context_x = context_x[choice]
            context_y = context_y[choice]
        if self.opt.shots != -1 and context_x.shape[0] > self.opt.shots:
            choice = np.random.choice(
                context_x.shape[0], self.opt.shots, replace=False)
            context_x = context_x[choice]
            context_y = context_y[choice]
        if self.split == 'train' and self.opt.points_target == True:
            target_x = np.array(context_x)
            target_y = np.array(context_y)
        else:
            target_x = self.all_x[index]
            target_y = self.all_y[index]
        #context_y   = np.linalg.norm(context_x, axis=-1)[...,None]-.5
        #target_y    = np.linalg.norm(target_x,  axis=-1)[...,None]-.5
        extra_data = dict(index=index,
                          vis=self.if_vis[index],
                          )
        # if self.split in ['val', 'test']:
        #     #extra_data['']
        #     pass
        item = dict(context_x=context_x,
                    context_y=context_y,
                    target_x=target_x,
                    target_y=np.nan_to_num(target_y),
                    **extra_data,
                    )
        return item

    def __len__(self):
        """Return the total number of images."""
        return self.length * self.multiplier


def partial_select(points, params):
    # param[0]: axis, param[1]:direction, param[2]:fraction
    axis, direction, frac = int(params[0]), int(params[1]), params[2]
    leng = points[:, axis].max() - points[:, axis].min()
    select_leng = leng * frac
    if direction == 0:
        lim = points[:, axis].min()
        sign = 1
        bound = (lim, lim + select_leng)
    else:
        lim = points[:, axis].max()
        sign = -1
        bound = (lim - select_leng, lim)
    return np.logical_and(points[:, axis] >= bound[0], points[:, axis] < bound[1])

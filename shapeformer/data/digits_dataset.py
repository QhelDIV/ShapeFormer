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


class DigitsDataset(BaseNPDataset):
    def default_opt(self):
        return {
            'datah5':       'digits_final.h5',
            'exclude': 8,  # [0,10) or [0,20) depending on exclude_mode
            'exclude_mode': 'digit',
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
        if self.opt.exclude != -1:
            array = np.zeros(200, np.int)
            if self.opt.exclude_mode == 'style':
                array[self.opt.exclude + 20*np.arange(10)] = 1
            elif self.opt.exclude_mode == 'digit':
                array[np.arange(self.opt.exclude*20,
                                (self.opt.exclude+1)*20)] = 1
            self.split_train = np.where(array == 0)[0]
            self.split_test = np.where(array == 1)[0]
        self.select = self.split_train if split == 'train' else self.split_test
        self.x = self.all_x[self.select]
        self.y = self.all_y[self.select]
        # total_num = self.select.shape[0]
        # self.length = 800 if split=='train' else 200
        # self.split_indices = np.random.randint(0, total_num, self.length)
        # self.x = self.all_x[self.split_indices]
        # self.y = self.all_y[self.split_indices]
        # self.points = self.points[self.split_indices]
        #self.perm = np.arange(self.length)
        #print('Dataset initialized')
        self.length = self.x.shape[0]

    def __getitem__(self, index):
        context_x = self.points[index]
        context_y = np.zeros((context_x.shape[0], self.opt.dim_y))
        if self.split == 'train' and hasattr(self.opt, 'points_target') and self.opt.points_target == True:
            target_x = np.array(context_x)
            target_y = np.array(context_y)
        else:
            target_x = self.x[index]
            target_y = self.y[index]
        item = dict(context_x=context_x,
                    context_y=context_y,
                    target_x=target_x,
                    target_y=target_y,
                    )
        return item

    def sample(self):
        id = np.random.randint(self.length)
        return self.__getitem__(id)

    def __len__(self):
        """Return the total number of images."""
        return self.length


class LegacyDigitsDataset(BaseNPDataset):

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
        self.collate_fn = datautils.NP_collate_fn
        self.datapath = os.path.join(opt.datasets_dir, opt.datah5)
        with h5py.File(self.datapath, 'r') as f:
            self.all_x = np.array(f['sdf_x'])
            self.all_y = np.array(f['sdf_y'])
            #self.all_points = np.array(f['points'])
            self.split_train = np.array(f['split_train'])
            self.split_test = np.array(f['split_test'])
            if hasattr(opt, 'exclude') and opt.exclude != -1:
                self.split_train = np.concatenate(
                    [np.arange(0, opt.exclude*20), np.arange((opt.exclude+1)*20, 200)])
                self.split_test = np.arange(opt.exclude*20, (opt.exclude+1)*20)
        total_num = self.all_x.shape[0]
        select = self.split_train if split == 'train' else self.split_test
        num = 800 if split == 'train' else 200
        self.split_indices = np.random.randint(0, 200, num)
        self.x = self.all_x[self.split_indices]
        self.y = self.all_y[self.split_indices]
        # if hasattr(opt, 'binary') and opt.binary==True:
        #    self.y = np.sign(self.y)

        self.length = self.x.shape[0]
        #self.perm = np.arange(self.length)
        #print('Dataset initialized')

    def __len__(self):
        """Return the total number of images."""
        return self.length

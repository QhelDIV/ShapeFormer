import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

from shapeformer.data.sdf_dataset import SDFDataset
from shapeformer import vis
from shapeformer.util import util
from shapeformer.data import utils as datautils


class NNReconDataset(SDFDataset):
    def __init__(self, opt, dataDict=None, split='train'):
        # save the option and dataset root
        # sdf_x,sdf_y,points,split_train,split_test
        super().__init__(opt, dataDict, split)
        self.collate_fn = datautils.nnrecon_collate_fn

        self.all_points = util.H5Var(self.datapath, 'points')[None]
        # ,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]]#self.split_indices]
        self.points = self.all_points[select_ind]
        self.x = self.all_x[select_ind]
        self.y = self.all_y[select_ind]

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.
        """
        index = index % self.points.shape[0]
        #self.all_points[index] = np.array([[1.,0.],[0.,1.],[-1.,0.]])*.4
        #self.all_points[index] = np.array([[1.,0.],[.5,.5],[0.,1.],[-.5,.5],[-1.,0.],[-.5,-.5],[0.,-1.],[.5,-.5]]*100)*.8
        context_x = self.points[index]
        context_y = np.zeros((context_x.shape[0], 1))
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

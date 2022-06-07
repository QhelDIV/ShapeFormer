'''


'''
from shapeformer import options
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

from shapeformer.util import util, geoutil
from shapeformer.data import utils as datautils
from shapeformer import vis
from shapeformer.data.basenp_dataset import BaseNPDataset


class BasicGeomDataset(BaseNPDataset):
    '''
    '''

    def default_opt(self):
        return {
            'datah5':       'box_3D.h5',
            'shots':        256,
            'multiplier':   1,
            'vis_indices':  [],
            # 'points_target':False,
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
        print(self.datapath)
        self.dataDict = util.readh5(self.datapath)
        self.select = self.dataDict['split_train' if split ==
                                    'train' else 'split_test']
        self.all_points = self.dataDict['points']
        self.length = self.select.shape[0]
        self.parse_vis_index(length=self.length)

    def __getitem__(self, index):
        index = index % self.length
        ifvis = self.if_vis[index]
        index = self.select[index]
        context_x = self.dataDict['context_x']  # [index]
        context_y = np.zeros((context_x.shape[0], 1))
        target_x = self.dataDict['target_x']
        target_y = self.dataDict['target_y'][index]
        extra_data = dict(index=index,
                          vis=ifvis,
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
        return self.length * self.opt.multiplier


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


def generate_box_dataset(name='', dim_x=3, boxspec=np.array([[.3, .7], [.3, .7], [.3, .7]]),
                         grid_dim=32):
    total_num = 400
    split_train = np.arange(0, total_num//10*9)
    split_test = np.arange(total_num//10*9, total_num)
    print(split_test)
    dim_x = 3

    lenspec = boxspec[:, 1] - boxspec[:, 0]
    anchor_point = np.array([-.8, -.8, -.8])

    target_x = util.makeGrid(bb_min=(-1,)*dim_x, bb_max=(1,)*dim_x,
                             shape=(grid_dim,)*dim_x
                             )
    spec = np.random.rand(total_num, 3)
    spec = spec % (boxspec[:, 1]-boxspec[:, 0]
                   )[None, ...] + boxspec[:, 0][None, ...]

    center = anchor_point - (-spec)
    target_ys = geoutil.batchBoxSDF(target_x, spec, center)
    complete_points = []
    for i in util.progbar(range(target_ys.shape[0])):
        vert, face = geoutil.array2mesh(
            target_ys[i], coords=target_x, dim=dim_x)
        complete_points.append(geoutil.sampleMesh(vert, face, 8192))
    complete_points = np.array(complete_points)
    context_x = complete_points[0]
    context_x = context_x[context_x.sum(axis=-1)/np.sqrt(3) < -1.1]

    target_path = os.path.join(options.datasets_dir, f'box_{dim_x}D{name}.h5')
    dataDict = {'points': complete_points,
                'context_x': context_x,
                'target_x': target_x,
                'target_y': target_ys[..., None],
                'split_train': split_train,
                'split_test': split_test,
                }
    util.writeh5(target_path, dataDict)
    return dataDict

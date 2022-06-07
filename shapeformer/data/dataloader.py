import sys
import random
import functools
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from scipy.stats import betabinom

from xgutils import sysutil

thismodule = sys.modules[__name__]
__all__ = [
    "AllSelector",
    "NoneSelector",
    "CornerSelector",
    "UniformSubsampleSelector",
    "HalfSelector",
    "CtTgCollator",
    "PCCollator",
]

def ratio2int(percentage, max_val):
    if type(percentage) is float:
        if 0. <= percentage <= 1.:
            out = percentage * max_val
        else:
            raise ValueError(f'float percentage must lay in [0,1], input percentage:{percentage}')
    elif type(percentage) is int:
        if max_val < percentage or percentage < 0:
            out = max_val
        elif 0 <= percentage <= max_val:
            out = percentage
    return out
def parse_range(r, max_num):
    if type(r) is int:
        if r==-1:
            num = max_num
        else:
            raise ValueError('%s should be -1'%r)
    else:
        r=list(r)
        assert(len(r)==2)
        r[0] = ratio2int(r[0], max_num)
        r[1] = ratio2int(r[1], max_num) + 1
        num = np.random.randint(*r)
    return num
def random_choice(total, choiceN, sort=False):
    locations = np.random.choice( \
                    total,
                    size=choiceN,
                    replace=False)
    locations = locations if sort==False else np.sort(locations)
    return locations
class AllSelector():
    def __call__(self, X, Y):
        return X, Y
class CornerSelector():
    def __init__(self, corner='lower'):
        self.corner = corner
    def __call__(self, X, Y):
        inds = X.sum(axis=2).argmin(axis=1)
        subX = torch.stack([X[i,ind][None,...] for i,ind in enumerate(inds)])
        subY = torch.stack([Y[i,ind][None,...] for i,ind in enumerate(inds)])
        return subX, subY
class HalfSelector():
    def __init__(self, portion, portion_on='cardinality', axis=-1, descending=True):
        """select portion of context_set
            If select by distance, then select_num will always be total_num

        Args:
            portion (float): float in [0,1]
            portion_on (str, optional): ['cardinality', 'distance'] base on point set cardinality or absolution shape
            orientation (str, optional): ['top','down','left','right']. Defaults to 'top'.
        """
        self.portion, self.portion_on = portion, portion_on
        self.descending, self.axis = descending, axis
        if portion_on != 'cardinality' and portion_on != 'distance':
            raise NotImplementedError("Only portion_on cardinality or distance is supported now")
    def __call__(self, X, Y):
        batch_size, total_size, dim_x = X.shape
        if type(self.portion) is float:
            portion = self.portion
        elif type(self.portion) is list or type(self.portion) is tuple:
            portion = self.portion[0] + (self.portion[1]-self.portion[0])*np.random.rand()
        
        dim_y = Y.shape[-1]
        coord = X[:,:,self.axis]
        if self.portion_on == 'cardinality':
            select_num = int(total_size * portion)
            ind = coord.sort(axis=-1, descending=self.descending)[1][:,:select_num]
        elif self.portion_on == 'distance':
            select_num = total_size
            ind = np.zeros((batch_size, select_num))
            for i in range(batch_size):
                index_i = coord[i,:] >= portion if self.descending==True else coord[i,:] <= portion
                index_i = torch.nonzero(index_i)[:,0]
                if index_i.shape[0]==0:
                    print("Empty set encountered, regard all data in this batch as empty!")
                    return torch.zeros(batch_size, 0, dim_x), torch.zeros(batch_size, 0, dim_y)
                ind[i] = index_i[ np.random.choice(index_i.shape[0], select_num, replace=True) ]
            ind = torch.tensor(ind,dtype=int)

        indX = ind.unsqueeze(-1).expand(-1, -1, dim_x)
        indY = ind.unsqueeze(-1).expand(-1, -1, dim_y)
        subX = X.gather(1, indX)
        subY = Y.gather(1, indY)
        return subX, subY
    def unittest(verbose=False, **kwargs):
        gen = HalfSelector(**kwargs)
        X=torch.tensor([ [[1,4], [3,-1], [2,5]],
                         [[4,4], [5,6],  [1,0]]]) # 2x3x2
        Y=torch.tensor([ [[-1],[-2],[2]],
                         [[-3],[-4],[5]]]) # 2x3x1
        sX,sY = gen(X,Y)
        if verbose:
            print('sX:',sX.shape)
            print(sX)
            print('sY:', sY.shape)
            print(sY)
        return True
class NoneSelector():
    def __call__(self, X, Y):
        batch_size, total_size, dim_x = X.shape
        dim_y = Y.shape[-1]
        NoneX = torch.zeros( (batch_size, 0, dim_x) )
        NoneY = torch.zeros( (batch_size, 0, dim_y) )
        return NoneX, NoneY

class UniformSubsampleSelector():
    def __init__(self, interval):
        self.interval   = interval
    def __call__(self, X, Y):
        batch_size, total_size, dim_x = X.shape
        select_num = parse_range(self.interval, total_size)
        dim_y = Y.shape[-1]
        ind = np.zeros((batch_size, select_num))
        for i in range(batch_size):
            ind[i] = np.random.choice(total_size, select_num, replace=False)
        indX = torch.tensor(ind,dtype=int).unsqueeze(-1).expand(-1, -1, dim_x)
        indY = torch.tensor(ind,dtype=int).unsqueeze(-1).expand(-1, -1, dim_y)
        subX = X.gather(1, indX)
        subY = Y.gather(1, indY)
        return subX, subY
    def unittest(verbose=False):
        gen = UniformSubsampleSelector(interval=(1,2))
        X=torch.tensor([    [[1,1,1],[2,2,2],[-2,-2,-2]],
                            [[3,3,3],[4,4,4],[-5,-5,-5]]
                            ])
        Y=torch.tensor([[[-1],[-2],[2]],[[-3],[-4],[5]]])
        sX,sY = gen(X,Y)
        if verbose:
            print('sX:',sX.shape)
            print(sX)
            print('sY:', sY.shape)
            print(sY)
        return True
class CtTgCollator():
    def __init__(self,  context_selector='UniformSubsampleSelector', ct_kwargs={}, 
                        target_selector='AllSelector',  tg_kwargs={}):
        self.ctGen = getattr(thismodule, context_selector)(**ct_kwargs)
        self.tgGen = getattr(thismodule, target_selector)(**tg_kwargs)
    def __call__(self, batch): # the function formerly known as "bar"
        # collate_fn
        batch = default_collate(batch)
        batch['Xct'], batch['Yct'] = self.ctGen(batch['context_x'], batch['context_y'])
        batch['Xtg'], batch['Ytg'] = self.tgGen(batch['target_x'],  batch['target_y'])
        del batch['context_x'], batch['context_y'], batch['target_x'], batch['target_y']
        for key in batch:
            batch[key] = batch[key].float()
        return batch
class pc_selector_wrapper():
    def __init__(self, selector):
        self.selector = selector
    def __call__(self, X):
        Y = torch.zeros_like(X)
        subX, subY = self.selector(X, Y)
        return subX
class PCCollator():
    def __init__(self,  context_selector='UniformSubsampleSelector', ct_kwargs={}, 
                        target_selector='AllSelector',  tg_kwargs={}):
        self.ctGen = pc_selector_wrapper(getattr(thismodule, context_selector)(**ct_kwargs))
        self.tgGen = getattr(thismodule, target_selector)(**tg_kwargs)
    def __call__(self, batch): # the function formerly known as "bar"
        # collate_fn
        batch = default_collate(batch)
        batch['Xbd'] = batch['context_x']
        batch['Xct'] = self.ctGen(batch['Xbd'])
        batch['Xtg'], batch['Ytg'] = self.tgGen(batch['target_x'],  batch['target_y'])
        del batch['context_x'], batch['context_y'], batch['target_x'], batch['target_y']
        for key in batch:
            batch[key] = batch[key].float()
        return batch
    

if __name__ == '__main__':
    unittest()

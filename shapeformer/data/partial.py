import sys
import random
import functools
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
    
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from scipy.stats import betabinom
from xgutils import *

thismodule = sys.modules[__name__]
# __all__ = [
#     "AllSelector",
#     "NoneSelector",
#     "CornerSelector",
#     "UniformSubsampleSelector",
#     "HalfSelector",
#     "CtTgCollator",
#     "PCCollator",
# ]

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
    def __init__(self, context_N=None):
        self.context_N=context_N
    def __call__(self, Xbd):
        Xct = Xbd
        if self.context_N is not None:
            choice = np.random.choice(Xct.shape[0], self.context_N, replace=True)
            Xct = Xct[choice]
        return Xct
class CornerSelector():
    def __init__(self, corner='lower'):
        self.corner = corner
    def __call__(self, Xbd, **kwargs):
        inds = Xbd.sum(axis=2).argmin(axis=1)
        Xct = Xbd[ind][None,...]
        return Xct
class BallSelector():
    def __init__(self, radius=.1, context_N=512, noise=0., inverse=False):
        self.noise = noise
        self.inverse = inverse
        self.radius, self.radius2 = radius, radius**2
        self.context_N = context_N
    def __call__(self, Xbd, radius=None, **kwargs):
        p_ind = np.random.choice(Xbd.shape[0], 1)
        pivot = Xbd[p_ind]
        dist = ((Xbd - pivot)*(Xbd-pivot)).sum(axis=-1)
        rd2 = self.radius2 if radius is None else radius**2
        selection = (dist < rd2)
        if self.inverse==True:
            selection = np.logical_not(selection)
            if selection.sum()<400:
                selection = np.ones_like(selection).astype(bool)
        Xct = Xbd[selection]
        if self.context_N>=0:
            choice = np.random.choice(Xct.shape[0], self.context_N, replace=True)
            Xct = Xct[choice]
        if self.noise>0:
            Xct += np.random.randn(*Xct.shape)*self.noise
            Xct = Xct.clip(-1., 1.)
        return Xct
class MultiBallSelector():
    def __init__(self, radius_range=[.05, .4], N_range=[1,3], context_N=512, virtual_scan=False):
        self.radius_range = radius_range
        self.N_range  = N_range
        self.context_N = context_N
        self.virtual_scan=virtual_scan
        self.selector = BallSelector(context_N=context_N)
    def __call__(self, Xbd):
        N = np.random.randint(*self.N_range)
        rr = self.radius_range
        Xct = []
        if self.virtual_scan==True:
            nXbd = geoutil.hidden_point_removal(Xbd, geoutil.sample_sphere(1)[0]*10 )
            if nXbd.shape[0]<=2:
                print("warning, virtual scanned points less than 2")
                print("Use Xbd as Xct")
                nXbd = Xbd
            Xbd = nXbd

        for i in range(N):
            radius = rr[0] + np.random.rand()*(rr[1]-rr[0])
            Xct.append(self.selector(Xbd, radius=radius))
        Xct = np.concatenate(Xct, axis=0)
        choice = np.random.choice(Xct.shape[0], self.context_N, replace=True)
        Xct = Xct[choice]
        return Xct

class VirtualScanSelector():
    def __init__(self, radius=10, context_N=512, noise=0., manual_cameras={}):
        self.__dict__.update(locals())
    def __call__(self, Xbd, index=None, **kwargs):
        #if index in self.manual_cameras:
        #C = self.manual_cameras[179]
        #else:
        C = geoutil.sample_sphere(1)[0]*self.radius
        Xct = geoutil.hidden_point_removal(Xbd, C)
        if Xct.shape[0]<=2:
            print("warning, virtual scanned points less than 2")
            print("Use Xbd as Xct")
            Xct = Xbd

        if self.context_N>=0:
            choice = np.random.choice(Xct.shape[0], self.context_N, replace=True)
            Xct = Xct[choice]
        if self.noise>0:
            Xct += np.random.randn(*Xct.shape)*self.noise
            Xct = Xct.clip(-1., 1.)
        return Xct
class MixSelector_fixed1():
    def __init__(self, context_N=512):
        self.selectors = []
        self.selectors.append(VirtualScanSelector(context_N=context_N))
        self.selectors.append(OrthoVirtualScanSelector(context_N=context_N))
        self.selectors.append(AllSelector(context_N=context_N))
        self.selectors.append(MultiBallSelector(context_N=context_N))
    def __call__(self, Xbd, **kwargs):
        choice = np.random.randint(0, len(self.selectors))
        #print("***",choice)
        Xct = self.selectors[choice](Xbd)
        return Xct
class OrthoVirtualScanSelector():
    def __init__(self, radius=10, context_N=512, noise=0.):
        self.__dict__.update(locals())
    def __call__(self, Xbd, **kwargs):
        chosen_axis = np.random.choice(Xbd.shape[-1], 1)[0]
        chosen_dir  = np.random.choice(2,1)[0]*2 - 1
        pos = np.zeros(Xbd.shape[-1])
        pos[chosen_axis] = 1 * chosen_dir
        #print(pos)
        C = pos*self.radius
        Xct = geoutil.hidden_point_removal(Xbd, C)
        if Xct.shape[0]<=2:
            print("warning, virtual scanned points less than 2")
            print("Use Xbd as Xct")
            Xct = Xbd

        if self.context_N>=0:
            choice = np.random.choice(Xct.shape[0], self.context_N, replace=True)
            Xct = Xct[choice]
        if self.noise>0:
            Xct += np.random.randn(*Xct.shape)*self.noise
            Xct = Xct.clip(-1., 1.)
        return Xct
class CamVirtualScanSelector():
    def __init__(self, radius=10, context_N=512, noise=0.):
        self.__dict__.update(locals())
    def __call__(self, Xbd, camera_pos, **kwargs):
        C = camera_pos
        Xct = geoutil.hidden_point_removal(Xbd, C)
        if Xct.shape[0]<=2:
            print("warning, virtual scanned points less than 2")
            print("Use Xbd as Xct")
            Xct = Xbd

        if self.context_N>=0:
            choice = np.random.choice(Xct.shape[0], self.context_N, replace=True)
            Xct = Xct[choice]
        if self.noise>0:
            Xct += np.random.randn(*Xct.shape)*self.noise
            Xct = Xct.clip(-1., 1.)
        return Xct

class fixedVirtualScanSelector():
    def __init__(self, radius=10, context_N=512, noise=0.):
        self.__dict__.update(locals())
    def __call__(self, Xbd, **kwargs):
        # C = geoutil.sample_sphere(1)[0]*self.radius
        C = [1,1,1]*self.radius
        Xct = geoutil.hidden_point_removal(Xbd, C)
        if Xct.shape[0]<=2:
            print("warning, virtual scanned points less than 2")
            print("Use Xbd as Xct")
            Xct = Xbd

        if self.context_N>=0:
            choice = np.random.choice(Xct.shape[0], self.context_N, replace=True)
            Xct = Xct[choice]
        if self.noise>0:
            Xct += np.random.randn(*Xct.shape)*self.noise
            Xct = Xct.clip(-1., 1.)
        return Xct

class HalfSpaceSelector():
    def __init__(self, portion, portion_on:Literal["cardinality", "distance"] ="cardinality", context_N=512, 
    plane_normal=[1,0,0.], plane_o = [0,0,0.]):
        """Divide the 3D shape with a plane and select the half space containing plane normal `plane_normal`
            select by distance or cardinality according to `portion_on` 

        Args:
            portion (float): float in [0,1], select what percentage of the object
            portion_on (str, optional): ['cardinality', 'distance'] base on point set cardinality or absolution shape
            plane_normal (list, optional): ['top','down','left','right']. Defaults to [1,0,0], that is +x.
        """
        self.__dict__.update(locals())
        self.plane_o      = np.array(self.plane_o)
        self.plane_normal = np.array(self.plane_normal)
        self.plane_normal = nputil.normalize(self.plane_normal)
    def __call__(self, Xbd, **kwargs):
        total_size, dim_x = Xbd.shape
        portion = self.portion
        distance = ( (Xbd-self.plane_o[None,...]) * self.plane_normal[None,...]).sum(axis=-1)
        if self.portion_on == 'cardinality':
            select_num = int(total_size * portion)
            selection = np.argsort(distance, descending=True)[:select_num]
        elif self.portion_on == 'distance':
            select_num = total_size
            max_diff = distance.max() - distance.min()
            threshold = max_diff*portion + distance.min()
            selection = np.where( distance>=threshold )[0]
        if selection.shape[0]>0:
            choice = np.random.choice(selection.shape[0], self.context_N, replace=True)
            selection = selection[choice]
        return Xbd[selection]
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

# TODO need modification.
class HalfSelector_torch():
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
        total_size, dim_x = X.shape
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

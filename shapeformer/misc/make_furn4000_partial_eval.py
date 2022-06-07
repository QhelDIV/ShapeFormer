from shapeformer.util import util
import sys
import h5py
import numpy as np
sys.path.append('.')


def test():
    data = h5py.File('datasets/furn4000.h5', 'r')
    print(data.keys())
    data.close()
    data = util.readh5('datasets/furn4000.h5')
    print(data.keys())


def make_partial_params():
    with h5py.File('datasets/furn4000.h5', 'r') as f:
        leng = f['points'].shape[0]
    axis = np.random.randint(0, 3, leng)
    direction = np.random.randint(0, 2, leng)
    bound = [.2, .4]
    frac = (bound[1]-bound[0]) * np.random.rand(leng) + bound[0]
    params = np.stack([axis, direction, frac], axis=-1)
    with h5py.File('datasets/furn4000.h5', 'r+') as f:
        f['partial_params'] = params


if __name__ == '__main__':
    make_partial_params()

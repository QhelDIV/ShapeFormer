import sys
sys.path.append('../')
import h5py
import numpy as np
import util.util as util
import os

def process(curvepath, sdfpath, outpath):
    # raw dataset contains:
    # ['char', 'edge', 'fonts', 'lengths', 'vertex']
    curveDict = util.readh5(curvepath)
    sdfDict   = util.readh5(sdfpath)

    sample = sdfDict['sample']
    label  = sdfDict['label']
    shapeId = sdfDict['shapeId']
    all_x = np.stack([sample[shapeId==i] for i in range(shapeId.max()+1)], axis=0)
    all_y = np.stack([label[shapeId==i]  for i in range(shapeId.max()+1)], axis=0)[..., np.newaxis]
    outDict = curveDict
    outDict.update({'sdf_x':all_x, 'sdf_y':all_y})
    outDict['points'] = outDict['vertex']
    total_shapes = all_x.shape[0]
    train_shapes = int(total_shapes*.1)
    test_shapes = total_shapes - train_shapes
    perm = np.random.permutation(total_shapes)
    outDict['perm'] = perm
    outDict['split_train'] = perm[:train_shapes]
    outDict['split_test']  = perm[train_shapes:]
    for key in outDict:
        print(key, outDict[key].shape)
    util.writeh5(outpath, outDict)
if __name__=='__main__':
    process(curvepath   = '/smartscan/nnrecon/datasets/digits/digits.h5', \
            sdfpath     = '/smartscan/nnrecon/datasets/digits/digits_sdf_legacy_format.h5', \
            outpath     = '/smartscan/nnrecon/datasets/digits/digits_final.h5')


import h5py
import igl
import numpy as np

from shapeformer.util import util

with h5py.File('/smartscan/scan2d/datasets/furn4000_setting.h5', 'r') as f:
    mesh_indices = np.array(f['index'])
all_mesh = util.H5File('/smartscan/scan2d/datasets/meshall.h5')

sampleN = 20480

verts = all_mesh["vert", mesh_indices]
faces = all_mesh["face", mesh_indices]
pcs = []
for i in util.progbar(range(verts.shape[0])):
    vert = verts[i]
    # face = faces[i]
    # B,FI = igl.random_points_on_mesh(sampleN, vert, face)
    # sampled =   B[:,0:1]*vert[face[FI,0]] + \
    #             B[:,1:2]*vert[face[FI,1]] + \
    #             B[:,2:3]*vert[face[FI,2]]
    # print(sampled.shape)

    # For uniform mesh, directly sample from vertices should be fine.
    pcs.append(vert[np.random.choice(vert.shape[0], sampleN, replace=False)])

pcs = np.array(pcs)
resample = False
if resample:
    SDFdataDict = util.readh5(
        '/smartscan/scan2d/datasets/furn4000_shapeSDF_resampled.h5')
else:
    SDFdataDict = util.readh5(
        '/smartscan/scan2d/datasets/furn4000_shapeSDF.h5')

# change scale & loc
dataDict = {}
n = int(np.ceil(SDFdataDict['SDF'].shape[1]**(1/3)))
dataDict['sdf_x'] = (SDFdataDict['SDF'][..., 0:3]*(n+1)/n - .5) * 2
dataDict['sdf_y'] = SDFdataDict['SDF'][..., 3:4]*(n+1)/n*2
dataDict['points'] = (pcs*(n+1)/n-.5)*2

perm = np.random.permutation(pcs.shape[0])
dataDict['perm'] = perm
num_train = pcs.shape[0]*8 // 10
dataDict['split_train'] = perm[:num_train]
dataDict['split_test'] = perm[num_train:]
dataDict['mesh_indices'] = mesh_indices

if resample:
    util.writeh5('/smartscan/nnrecon/datasets/furn4000_resampled.h5', dataDict)
else:
    util.writeh5('/smartscan/nnrecon/datasets/furn4000.h5', dataDict)

import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
from xgutils import *
from xgutils.vis import fresnelvis, visutil

sample_cloudR = 0.02
vis_camera = dict(camPos=np.array([2,2,2]), camLookat=np.array([0.,0.,0.]),\
                camUp=np.array([0,1,0]),camHeight=2,resolution=(256,256), samples=32)
lowvis_camera = dict(camPos=np.array([2,2,2]), camLookat=np.array([0.,0.,0.]),\
                camUp=np.array([0,1,0]),camHeight=2,resolution=(128,128), samples=8)
HDvis_camera  = dict(camPos=np.array([2,2,2]), camLookat=np.array([0.,0.,0.]),\
                camUp=np.array([0,1,0]),camHeight=2,resolution=(512,512), samples=256)

sample_cloudR = 0.02
pink_color = np.array([255, 0, 0])/256
gold_color = np.array([253, 204, 134])/256
gray_color = np.array([0.9, 0.9, 0.9])
result_color  = gray_color
partial_color = gold_color


def vis_VoxelXRay(voxel, axis=0, duration=10., target_path="/studio/temp/xray.mp4"):
    if(len(voxel.shape)==1):
        voxel = nputil.array2NDCube(voxel, N=3)
    imgs = []
    for i in sysutil.progbar(range(voxel.shape[axis])):
        nvox = voxel.copy()
        nvox[i:,:,:]=0
        voxv, voxf = geoutil.array2mesh(nvox.reshape(-1), thresh=.5, coords=nputil.makeGrid([-1,-1,-1],[1,1,1],[64,64,64], indexing="ij"))
        #voxv[]
        dflt_camera = fresnelvis.dflt_camera
        dflt_camera["camPos"]=np.array([2,2,2])
        dflt_camera["resolution"]=(256,256)
        img = fresnelvis.renderMeshCloud({"vert":voxv,"face":voxf}, **dflt_camera, axes=True)
        imgs.append(img)
    sysutil.imgarray2video(target_path, imgs, duration=duration)

def OctreePlot3D(tree, dim, depth, **kwargs):
    assert dim==2
    boxcenter, boxlen, tdepth = ptutil.ths2nps(ptutil.tree2bboxes(torch.from_numpy(tree), dim=dim, depth=depth))    
    maxdep = tdepth.max()
    renderer = fresnelvis.FresnelRenderer(camera_kwargs=dict(camPos=np.array([1.5,2,2]), resolution=(1024,1024)))#.add_mesh({"vert":vert, "face":face})
    for i in range(len(tdepth)):
        dep=tdepth[i]
        length = boxlen[i]
        bb_min = boxcenter[i]-boxlen[i]
        bb_max = boxcenter[i]+boxlen[i]
        lw=1+.5*np.exp(-dep)
        #rect = patches.Rectangle(corner, 2*length, 2*length, linewidth=lw, edgecolor=plt.cm.plasma(dep/maxdep), facecolor='none')
        renderer.add_bbox(bb_min=bb_min, bb_max=bb_max, color=plt.cm.plasma(dep/maxdep)[:3], radius=0.001*dep**1.5, solid=.0)
    img = renderer.render()
    return img

#def CloudPlot

def SparseVoxelPlot(sparse_voxel, depth=4, varying_color=False, camera_kwargs=dict(camPos=np.array([2,2,2]), resolution=(512,512))):
    resolution = camera_kwargs["resolution"]
    if len(sparse_voxel)==0:
        return np.zeros((resolution[0], resolution[1], 3))
    grid_dim = 2**depth
    box_len  = 2/grid_dim/2

    renderer = fresnelvis.FresnelRenderer(camera_kwargs=camera_kwargs)#.add_mesh({"vert":vert, "face":face})
    voxel_inds   = ptutil.unravel_index( torch.from_numpy(sparse_voxel), shape=(2**depth,)*3 )
    voxel_coords = ptutil.ths2nps(ptutil.index2point(voxel_inds, grid_dim=grid_dim))

    color = fresnelvis.gray_color
    percentage = np.arange(len(voxel_coords)) / len(voxel_coords)
    if varying_color==True:
        color = plt.cm.plasma(percentage)[...,:3]
    renderer.add_box(center=voxel_coords, spec=np.zeros((3))+box_len, color=color, solid=0.)
    img = renderer.render()
    return img
acc_unis = []
def IndexVoxelPlot(pos_ind, val_ind, val_max=1024, depth=4, 
        manual_color=None, distinctive_color=True, 
        camera_kwargs=dict(camPos=np.array([2,2,2]), resolution=(512,512)),
        **kwargs
        ):
    resolution = camera_kwargs["resolution"]
    if len(pos_ind)==0:
        return np.zeros((resolution[0], resolution[1], 3))
    grid_dim = 2**depth
    box_len  = 2/grid_dim/2

    renderer = fresnelvis.FresnelRenderer(camera_kwargs=camera_kwargs)#.add_mesh({"vert":vert, "face":face})
    voxel_inds   = ptutil.unravel_index( torch.from_numpy(pos_ind), shape=(2**depth,)*3 )
    voxel_coords = ptutil.ths2nps(ptutil.index2point(voxel_inds, grid_dim=grid_dim))
    if distinctive_color == False:
        color = plt.cm.Blues(val_ind/val_max)[..., :3]
    else:
        unique, inverse = np.unique(val_ind, return_inverse=True)
        acc_unis.append(unique)
        acc_uni_n = np.unique( np.concatenate(acc_unis) ).shape[0]
        print(acc_uni_n)
        print("num total uniques", acc_uni_n)
        print("num uniques", unique.shape[0])
        color = plt.cm.Blues(inverse/unique.shape[0])[..., :3]
    if manual_color is not None:
        color = manual_color
    renderer.add_box(center=voxel_coords, spec=np.zeros((3))+box_len, color=color, solid=0., **kwargs)
    img = renderer.render()
    return img
def CubePlot(coords, size, color=None, cmap = "Blues", 
            camera_kwargs=dict(camPos=np.array([2,2,2]), resolution=(512,512)),
            renderer = None,
            **kwargs
        ):
    resolution = camera_kwargs["resolution"]
    if coords.shape[0]==0:
        return np.zeros((resolution[0], resolution[1], 3))
    if renderer is None:
        renderer = fresnelvis.FresnelRenderer(camera_kwargs=camera_kwargs)#.add_mesh({"vert":vert, "face":face})
    if color is None:
        color = np.zeros(coords.shape[0])
    if len(color.shape)==1:
        color = plt.get_cmap(cmap)(color)[..., :3]
    renderer.add_box(center=coords, spec=np.zeros((3))+size, color=color, solid=0., **kwargs)
    img = renderer.render()
    return img


def meshCloudPlot(Ytg, Xbd):
    vert, face = geoutil.array2mesh(Ytg.reshape(-1), thresh=.5, coords=nputil.makeGrid([-1,-1,-1],[1,1,1],[64,64,64], indexing="ij"))
    
    vis_camera = dict(camPos=np.array([2,2,2]), camLookat=np.array([0.,0.,0.]),\
            camUp=np.array([0,1,0]),camHeight=2,resolution=(512,512), samples=16)
    img = fresnelvis.renderMeshCloud({"vert":vert, "face":face}, cloud=Xbd, cloudR=.01, **vis_camera)
    return img


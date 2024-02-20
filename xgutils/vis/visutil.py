from time import time
import os
import io
import h5py
import glob
import re
from time import time
from PIL import Image
import copy

import numpy as np
import torch
import scipy

import matplotlib as mpl
from matplotlib import offsetbox
import matplotlib.image as mpimg
import matplotlib.cm as cm
import matplotlib.colors as mpcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable, ImageGrid
from mpl_toolkits.mplot3d import Axes3D  # This import registers the 3D projection, but is otherwise unused.
import matplotlib.pyplot as plt

from xgutils import nputil as util
from xgutils import nputil, ptutil

div=8
marginIn=.3
marginOut=.4
sdfcolors_inside  = plt.cm.Spectral(  np.linspace(1., 1-marginIn, 512) )
#sdfcolors_inside[-div:,:2] = sdfcolors_inside[-div,:2]
sdfcolors_inside[-div:,:3] = np.zeros_like(sdfcolors_inside[-div,:3])+.4#sdfcolors_inside[-div,:2]
sdfcolors_outside = plt.cm.Spectral( np.linspace(marginOut, 0., 512) )
#sdfcolors_outside[:div,1:] = sdfcolors_outside[div,1:]#sdfcolors_outside[:,1:]*.7
sdfcolors_outside[:div,:3] = np.zeros_like(sdfcolors_outside[:div,:3])+.4#sdfcolors_outside[:,1:]*.7
sdf_colors = np.vstack((sdfcolors_inside, sdfcolors_outside))
sdf_cmap = mpl.colors.LinearSegmentedColormap.from_list('sdf_cmap',    sdf_colors)

tmagmaColors = plt.cm.Oranges( np.linspace(0., 1., 256) )
tmagmaColors[:,3] = util.logistic(np.linspace(0.,1.,256), x0=.3,k=10)
tmagmaColors[:40,3]=0
tmagma_cmap = mpl.colors.LinearSegmentedColormap.from_list('tmagma',    tmagmaColors)

rhotColors = plt.cm.gist_heat(  np.linspace(1., 0, 256) )
rhotColors = np.r_[np.zeros((40,4)),rhotColors]
rhot_cmap = mpl.colors.LinearSegmentedColormap.from_list('rhot',    rhotColors)

rRdYlBuColors  = plt.cm.RdYlBu(  np.linspace(1.,0., 512) )
rRdYlBu_cmap = mpl.colors.LinearSegmentedColormap.from_list('rRdYlBu',    rRdYlBuColors)
def get_random_color(size, color_dim=16):
    # 32**3 = 32767
    with nputil.temp_seed(314):
        color = np.zeros((size, 3))
        rd = np.random.randint(color_dim**3, size=(size))
        color[..., 0] = rd // (color_dim*color_dim) 
        color[..., 1] = rd % (color_dim*color_dim) // color_dim
        color[..., 2] = rd % (color_dim*color_dim) % color_dim
    fac = 256 / color_dim
    return (color*fac) / 256
def get_random_cmap(size):
    rd_cmap = mpl.colors.LinearSegmentedColormap.from_list('random',    get_random_color(size))
    return rd_cmap
#random_cmap = mpl.colors.LinearSegmentedColormap.from_list('random',    get_random_color(1024))
def blankImg(resolution=(256,256), format="RGBA"):
    return np.zeros( (*resolution, 4) )
def newFig(resolution=(400.,400.), tight=True):
    dpi = resolution[0]/4.
    fig     = plt.figure(figsize=(resolution[0]/dpi, resolution[1]/dpi), dpi=dpi, tight_layout=tight)
    #fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig
def newPlot(resolution=(400.,400.), projection=None, withMargin=False, tight=True, \
            fig=None, ax=None, plotrange=None, ticks=None, axes_off=False, color_range=None):
    dpi = resolution[0]/4.
    if ax is None:
        fig, ax = plt.subplots(figsize=(resolution[0]/dpi, resolution[1]/dpi), dpi=dpi, tight_layout=tight)
    else:
        fig = ax.figure
    #if fig is None:
    #    fig, ax = plt.subplots(figsize=(resolution[0]/dpi, resolution[1]/dpi), dpi=dpi, tight_layout=tight)
    #elif ax is None:
    #    ax = fig.add_axes()

    # set plotrange
    if plotrange is not None:
        if type(plotrange) is np.ndarray:
            ax.set_xlim(plotrange[0,0], plotrange[0,1])
            ax.set_ylim(plotrange[1,0], plotrange[1,1])
        elif type(plotrange) is str:
            if plotrange=='square':
                plotrange = np.array([[-1.,1.],[-1.,1.]])
            elif plotrange=='psquare':
                plotrange = np.array([[0.,1.],[0.,1.]])
            else:
                raise ValueError(f"plotrange {plotrange} is not supported.")
        else:
            raise TypeError(f"plotrange type {type(plotrange)} is not supported.")
    if ticks is None:
        ax.set_xticks([])
        ax.set_yticks([])
    if axes_off==True:
        ax.axis('off')
    if color_range is not None:
        ax.set_clim(color_range)
    return fig, ax
def newPlot3D(resolution=(400.,400.), projection='3d', withMargin=False, tight=True, fig=None, ax=None):
    dpi = resolution[0]/4.
    if fig is None:
        fig = plt.figure(figsize=(resolution[0]/dpi, resolution[1]/dpi), dpi=dpi, tight_layout=tight)
    if ax is None:
        ax = fig.add_subplot('111', projection=projection)
    #ax.set_aspect('equal')
    return fig, ax
def readImg(path):
    return mpimg.imread(path)
def saveFig(target, fig, dpi=None):
    # target can be output file path or a IO buffer
    if type(fig) is tuple: # if the input is fig=(fig, ax) then only keep fig
        fig = fig[0]
    fig.savefig(target, transparent=True, dpi=dpi)
def saveImg(target, img):
    mpimg.imsave(target, img)
    return img
def saveImgs(targetDir='./', baseName='out_', imgs=[]):
    if not os.path.exists(targetDir):
        util.mkdirs(targetDir)
    if not util.isIterable(imgs):
        imgs = [imgs]
    for i,img in enumerate(imgs):
        saveImg(os.path.join(targetDir, '%s%d.png'%(baseName,i)), img)
def saveFigs(targetDir='./', baseName='out_', figs=[]):
    imgs = figs2imgs(figs)
    saveImgs(targetDir, baseName=baseName, imgs=imgs)
def fig2img(fig, dpi=None, closefig=True):
    t0 = time()
    buf = io.BytesIO()
    saveFig(buf, fig, dpi=dpi) # save figure to buffer
    buf.seek(0)
    image = np.array(Image.open(buf)).astype(float)/255.
    buf.close()
    if closefig:
        plt.close(fig)
    return image
def figs2imgs(figs):
    if not util.isIterable(figs):
        figs = [figs]
    return list(map(fig2img, figs))
def showFig(fig):
    pass
def showImg(img, scale=1, title=None, cmap=None, resolution=None, vmin=0., vmax=1.):
    if resolution is not None:
        fig,ax = newPlot(resolution=resolution)
    else:
        resolution = (int(img.shape[1]*scale), int(img.shape[0]*scale))
        fig, ax = newPlot(resolution=resolution)
    ax.set_axis_off()
    if title is not None:
        fig.suptitle(title)

    ax.imshow(img, cmap, vmin=vmin, vmax=vmax)
    plt.show()
    return fig, ax
def imageGrid(imgs, shape=None, zoomfac=1):
    imgs = np.array(imgs)
    if zoomfac!=1:
        if len(imgs.shape)==4:
            imgs = scipy.ndimage.zoom( imgs,[1, zoomfac,zoomfac, 1], order=0)
        else:
            imgs = scipy.ndimage.zoom( imgs,[1, zoomfac,zoomfac], order=0)

    numFig, imgDim = len(imgs), imgs[0].shape
    if shape is None:
        shape = np.ceil(np.sqrt(numFig)).astype(int)
        shape = np.array([shape, shape])
        shape[0] = np.ceil(numFig / shape[1]).astype(int) # remove blank row(s)
    else:
        shape = np.array(shape).T
        assert len(shape)==2, "Shape must be 2 dimensional"
        assert shape[0]!=-1 or shape[1]!=-1, "(-1,-1) not understood"
        if shape[0]==-1:
            shape[0] = np.ceil(numFig/shape[1]).astype(int)
        if shape[1]==-1:
            shape[1] = np.ceil(numFig/shape[0].astype(int))
    blankImg = np.zeros_like(imgs[0])
    supp_blanks = np.array([blankImg]*(shape[0]*shape[1] - numFig))
    blank_num = (shape[0]*shape[1] - numFig)
    if blank_num>0:
        imgs = np.concatenate([imgs, supp_blanks], axis=0)
    grid = np.repeat(np.repeat(blankImg, shape[0], axis=0), shape[1], axis=1)
    for i in range(shape[0]):
        for j in range(shape[1]):
            grid[ i*imgDim[0]:(i+1)*imgDim[0], j*imgDim[1]:(j+1)*imgDim[1] ] = imgs[i*shape[1]+j]
    return grid
    # fig = newFig(resolution=(1000,1000))
    # #fig = plt.figure(figsize=(4., 4.))
    # grid = ImageGrid(fig, 111,  # similar to subplot(111)
    #                 nrows_ncols=(shape[0], shape[1]),  # creates 2x2 grid of axes
    #                 axes_pad=0.05,  # pad between axes in inch.
    #                 aspect=True,
    #                 )
    # for ax, im in zip(grid, imgs):
    #     # Iterating over the grid returns the Axes.
    #     ax.imshow(im)
    #     #ax.set_xticks([])
    #     #ax.set_yticks([])
    #     ax.axis('off')
    #saveFig('ohno.png', fig)
def figGrid(figs, shape=None):
    return imageGrid(figs2imgs(figs), shape=shape)
def addCBar(target, fig, ax, nbins=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(target, cax=cax)
    if nbins is not None:
        cbar.ax.locator_params(nbins=3)
    return cbar
def standalone_colorbar():
    pl = plt
    a = np.array([[0,1]])
    pl.figure(figsize=(.12, 3))
    img = pl.imshow(a, cmap="plasma")
    pl.gca().set_visible(False)
    cax = pl.axes([0.1, 0.2, 0.8, 0.6])
    cbar = pl.colorbar(orientation="vertical", cax=cax)
    cyticks = np.array([1e-5, 1e-2, 0.1, 0.5])
    cbar.set_ticks(cyticks**expr)
    cbar.set_ticklabels(cyticks)
    #plt.savefig("colorbar.pdf", bbox_inches='tight')
    plt.show()
class Visualizer():
    def __init__(self, sample, label, pred=None):
        pass
def rescale(target, reference=None, truncate=False):
    if reference is None:
        reference = target
    rescaled = ( target - (reference.max() + reference.min())/2 ) / (reference.max() - reference.min()) + .5
    if truncate==True:
        rescaled[rescaled>1.]=1.
        rescaled[rescaled<0.]=0.
    return rescaled

def plot1D(samples=None, values=np.array([0,1,2,3]), title="plot"):
    fig, ax = newPlot()
    if samples is None:
        ax.plot(range(len(values)), values)
    fig.suptitle(title)
    return fig, ax
#def densityPlot(x,y,z,ax=plt,cmap='rainbow', squareRange=True, colorbar=True, ticks=None, norm=None):
def densityPlot(samples, values, cmap=sdf_cmap, bounds=None, squareRange=True, plotrange=np.array([[0.,1.],[0.,1.]]), colorbar=True, ticks=None, norm=None, resolution=(400.,400.), fig=None, ax=None):
    """Generate density plot from samples

    Args:
        samples (np.ndarray): samples to interpolate from
        values (np.ndarray): samples
        cmap ([type], optional): [description]. Defaults to sdf_cmap.
        bounds ([type], optional): [description]. Defaults to None.
        squareRange (bool, optional): [description]. Defaults to True.
        plotrange ([type], optional): [description]. Defaults to np.array([[0.,1.],[0.,1.]]).
        colorbar (bool, optional): [description]. Defaults to True.
        ticks ([type], optional): [description]. Defaults to None.
        norm ([type], optional): [description]. Defaults to None.
        resolution (tuple, optional): [description]. Defaults to (400.,400.).
        fig ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    fig, ax = newPlot(resolution=resolution, fig=fig, ax=ax)
    # 2D samples with value
    z = values
    if bounds is None:
        vmin, vmax = z.min(), z.max()
    else:
        vmin, vmax = bounds
    if norm == 'log':
        norm = mpcolors.LogNorm(vmin=vmin, vmax=vmax)
    elif norm == 'sdf':
        vmin, vmax = min(vmin, -0.01), max(vmax,0.99)
        norm = mpcolors.DivergingNorm(vmin=vmin, vcenter=0., vmax=vmax)
    elif norm is None:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    if samples is None: # grid input
        zi = z
    else:
        x,y = samples[:,0], samples[:,1]
        if squareRange == True:
            #xi = yi = np.arange(0,1.002,0.002)
            pass
        if plotrange is None:
            xi = np.linspace( x.min(), x.max(), 500)
            yi = np.linspace( y.min(), y.max(), 500)
        else:
            xi = np.linspace(*plotrange[0], 500)
            yi = np.linspace(*plotrange[1], 500)
        xi,yi = np.meshgrid(xi,yi)
        zi = scipy.interpolate.griddata((x,y),z,(xi,yi), method='linear')

    im = ax.imshow(zi, origin='lower', cmap=cmap, norm=norm)
    if ticks is None:
        ax.set_xticks([])
        ax.set_yticks([])
    if colorbar==True:
            #divider = make_axes_locatable(plt.gca())
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
    plt.close(fig) #Don't show it if 
    return fig, ax
def diffDensityPlot(s1,l1,s2,l2,ax,cmap='rainbow', colorbar=True):
    xi = yi = np.arange(0,1.002,0.002)
    xi,yi = np.meshgrid(xi,yi)
    zi1 = scipy.interpolate.griddata((s1[:,0],s1[:,1]),l1,(xi,yi), method='linear')
    zi2 = scipy.interpolate.griddata((s2[:,0],s2[:,1]),l2,(xi,yi), method='linear')
    im = ax.imshow( np.abs(zi1-zi2), extent=(0,1,0,1),origin='lower', cmap=cmap)
    if colorbar==True:
        if ax is plt:
            divider = make_axes_locatable(plt.gca())
        else:
            divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    return im
def diffDensityPlots(opt, candidateIds=[1,165,187, 132], cmap='rainbow', colorbar=True):
    from data import create_dataset
    xi = yi = np.arange(0,1.002,0.002)
    xi,yi = np.meshgrid(xi,yi)
    samples, labels, zis = [], [], []
    dataset = create_dataset(opt)
    sid = dataset.dataloader.dataset.dataDict['shapeId']
    for candidateId in candidateIds:
        samples.append( dataset.dataloader.dataset.dataDict['sample'][sid==candidateId] )
        labels.append( dataset.dataloader.dataset.dataDict['label'][sid==candidateId]   )
        zis.append( scipy.interpolate.griddata((samples[-1][:,0],samples[-1][:,1]),labels[-1],(xi,yi), method='linear') )
    samples, labels, zis = np.array(samples), np.array(labels), np.array(zis)
    zi_mean = zis.mean(axis=0)
    
    fig, axes = plt.subplots(2, len(candidateIds)+1, figsize=(15,15))
    axes[1, 0].imshow(zi_mean, extent=(0,1,0,1), origin='lower', cmap=cmap)
    for i in range(samples.shape[0]):
        im = axes[1, i+1].imshow( np.abs(zi_mean - zis[i]), extent=(0,1,0,1),origin='lower', cmap=cmap)
        im = axes[0, i+1].imshow( zis[i], extent=(0,1,0,1),origin='lower', cmap=cmap)
    plt.setp(axes.flat, aspect=1.0, adjustable='box')
    fig.tight_layout()
    return im
def addColorbar2Plot(fig, ax, im, position='right'):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(position, size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    return cbar
def gradientPlot(x,y,z,ax=plt,cmap='rainbow', colorbar=True):
    xi = yi = np.arange(0,1.002,0.002)
    xi,yi = np.meshgrid(xi,yi)
    zi = scipy.interpolate.griddata((x,y),z,(xi,yi), method='linear')
    dx, dy = np.gradient(zi)
    zi = np.sqrt(dx**2+dy**2)
    im = ax.imshow(zi,extent=(0,1,0,1),origin='lower', cmap=cmap)
    if colorbar==True:
        if ax is plt:
            divider = make_axes_locatable(plt.gca())
        else:
            divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    return im
def simplePlot(samples, labels, preds=None, save_path=None):
    if preds is None:
       preds = np.zeros_like(labels)
    valueses=[labels, preds, np.abs(labels-preds)]
    figs = [densityPlot(samples, values=values) for values in valueses]
    grid = figGrid(figs, shape=(1,3))
    return grid

def linePlot(samples, labels=None, ax=plt, style='', c_ind=0, plotrange=np.array([[0.,1.],[0.,1.]]), ticks=None):
    """Plot line (same as plt.plot, but with my preferred settings)

    Args:
        samples ([type]): [description]
        labels ([type], optional): [description]. Defaults to None.
        ax ([type], optional): [description]. Defaults to plt.
        style (str, optional): [description]. Defaults to ''.
        c_ind (int, optional): color index(mod by 20)(which color to use in tab20c). Defaults to 0.
        plotrange ([type], optional): [description]. Defaults to np.array([[0.,1.],[0.,1.]]).
        ticks ([type], optional): [description]. Defaults to None.
    """
    ax.plot(samples[:,0], samples[:,1], style, c = plt.cm.tab20(c_ind%20))
    if ax is plt:
        ax = plt.gca() #.set_aspect('equal', adjustable='box')
    ax.set_xlim(plotrange[0,0], plotrange[0,1])
    ax.set_ylim(plotrange[1,0], plotrange[1,1])
    ax.set_aspect('equal', adjustable='box' )
    if ticks is None:
        ax.set_xticks([])
        ax.set_yticks([])
    return ax.figure, ax

def scatterPlot(samples, labels=None, point_size = 3., cmap='rainbow', plotrange=np.array([[0.,1.],[0.,1.]]), ticks=None, colorbar=False, **kwargs):
    fig, ax = newPlot(**kwargs)
    im = ax.scatter(samples[:,0], samples[:,1], c=labels, s=point_size, cmap=cmap)
    #if ax is plt:
    #    ax = plt.gca() #.set_aspect('equal', adjustable='box')
    ax.set_xlim(plotrange[0,0], plotrange[0,1])
    ax.set_ylim(plotrange[1,0], plotrange[1,1])

    if colorbar==True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.locator_params(nbins=3)

    return fig, ax

def SDFPlotData(samples, labels=None, cmap='rainbow'):
    if labels is None:
        labels = samples[:,1]
    #print(labels.dtype,labels.shape)
    surfacePts = ((labels< .03) & (labels >=-.03))
    otherPts   = util.subsampleBoolArray(surfacePts, 4000)
    surfacePts = util.subsampleBoolArray(~surfacePts, 6000) # maximum 10000 points in the plot
    insidePts  = ((labels< .01) & otherPts)
    outsidePts = ((~insidePts) & otherPts)
    mask = (surfacePts | otherPts)
    labels[insidePts] = labels[outsidePts].min()
    scL = rescale(labels[mask],labels[mask])
    #filted= np.random.choice(ret.shape[0], 5000, replace=False)

    cmapF = mpl.cm.get_cmap(cmap)
    rgba_colors = cmapF(scL)
    rgba_colors[:, 3] = (1- np.power(scL, .1))*.76+.24
    print(rgba_colors[:, 3].min(), rgba_colors[:, 3].max())
    sizes = rgba_colors[:, 3]*10
    return mask, rgba_colors, sizes
def SDF3DPlot(samples, labels=None, rgba_colors=None, sizes=None, cmap='rainbow', title=None, ptsScale=.5, plotrange=np.array([[0.,1.],[0.,1.],[0.,1.]]), resolution=(400.,400.), fig=None, ax=None):
    fig, ax = newPlot3D(resolution=resolution, projection='3d',fig=fig, ax=ax)
    mask = np.ones(samples.shape[0], dtype=bool)
    if rgba_colors is None:
        mask, rgba_colors, sizes = SDFPlotData(samples, labels, cmap)
    ax.scatter(samples[mask,0], samples[mask,1], samples[mask,2], c=rgba_colors, s=sizes*ptsScale)#, mpl.rcParams['lines.markersize'] ** 2/5.)
    if title is None:
        title = "min:%.3f max:%.3f"%(labels.min(), labels.max())
    fig.suptitle(title)
    return fig, ax
def SDF2DPlot(gridData, pts=None, scatterSize=6, cmap=rRdYlBu_cmap, contour_mode=True, colorbar=True, norm=None, ticks=None, bbox=None, plotrange=np.array([[0.,1.],[0.,1.]]), valuerange=None, resolution=(400.,400.), fig=None, ax=None):
    fig, ax = newPlot(resolution=resolution, fig=fig, ax=ax)
    zoom_factor = 3
    gridDim = gridData.shape
    Z = scipy.ndimage.zoom(gridData, zoom_factor, order=1)
    if bbox is None:
        bbox = plotrange.T
    X, Y = np.meshgrid( np.linspace(bbox[0,0], bbox[1,0], gridDim[0]*zoom_factor), \
                        np.linspace(bbox[0,1], bbox[1,1], gridDim[1]*zoom_factor))
    
    vmin = min(Z.min(), -0.0001)
    vmax = max(Z.max(), 0.0001)

    if contour_mode==True:
        contour = ax.contour(X,Y,Z, levels=[0.], colors=('k',) ,linestyles=('-',), linewidths=(2,))
        #contour = ax.contour(X,Y,Z, colors=('k',) ,linestyles=('-',), linewidths=(1,))
    if ticks is None:
        ax.set_xticks([])
        ax.set_yticks([])
    if contour_mode == True:
        field = ax.contourf(X,Y,Z, cmap=rRdYlBu_cmap, norm = mpcolors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax))
    else:
        field = ax.imshow(np.flip(Z,axis=0), cmap=rRdYlBu_cmap,  norm = mpcolors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax))
    #print(contourF)
    if colorbar==True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(field, cax=cax, format=mpl.ticker.FuncFormatter(lambda x,pos:'%.1f'%x))
    # if valuerange is not None:
    #     mi = valuerange[0]
    #     ma = valuerange[1]
    #     contourF.set_clim(mi, ma)
    if pts is not None:
        pc = ax.scatter(pts[:,0], pts[:,1], s=scatterSize, marker='o', c='b')
    ax.set_aspect('equal')
    return fig, ax
def FieldPlot(gridData, pts=None, extent=[[-1.,1.],[-1.,1.]], scatterSize=6, zoomfac=3., \
        cmap='Reds', valuerange=np.array([-1.,1.]), colorbar=True, **kwargs):
    fig, ax = newPlot(**kwargs)
    Z = scipy.ndimage.zoom(gridData, zoomfac, order=1)

    field = ax.imshow(np.flip(Z,axis=0), cmap=cmap, extent=[extent[0][0], extent[0][1], extent[1][0], extent[1][1]])
    field.set_clim(valuerange)
    #print(contourF)
    if colorbar==True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(field, cax=cax, format=mpl.ticker.FuncFormatter(lambda x,pos:'%.1f'%x))
        cbar.ax.locator_params(nbins=3)
    return fig, ax
def labels2rgba(labels, cmap='rainbow', return_rescaled=False, noRescale=False):
    if labels.max() - labels.min() <0.000001:
        scL = np.zeros_like(labels)
    else:
        if noRescale==True:
            scL = np.clip(labels, 0., 1.)
        else:
            scL = rescale(labels,labels)
    if type(cmap) is str:
        cmapF = mpl.cm.get_cmap(cmap)
    else:
        cmapF = cmap
    rgba_colors = cmapF(scL)
    # too_black = rgba_colors<.5
    # rgba_colors[too_black] = (rgba_colors[too_black]*2.*.7+.3)/2.
    if return_rescaled==True:
        return rgba_colors, scL
    return rgba_colors
def scatter3D(samples, labels=None, noRescale=False, maxPts=6000, ptsSize=6, axis=False, cmap='rainbow', alpha=1, title=None, plotrange=np.array([[0.,1.],[0.,1.],[0.,1.]]), resolution=(400.,400.), fig=None, ax=None):
    fig, ax = newPlot3D(resolution=resolution, projection='3d',fig=fig, ax=ax)
    all_pts = util.subsampleBoolArray(np.ones(samples.shape[0],dtype=bool), maxPts)
    if labels is None:
        #labels = samples[:,2]+samples[:,1]+samples[:,0]
        labels = samples[:,0] + 10*samples[:,1] + 3*samples[:,2] # color accoring to y 
    rgba_colors, rescaled = labels2rgba(labels[all_pts],cmap, return_rescaled=True, noRescale=noRescale)
    # print(rgba_colors[0])
    rgba_colors[...,3] *= alpha
    #rgba_colors[:,3] = 1 - rescaled*rescaled

    
    ax.set_xlim(plotrange[0,0], plotrange[0,1])
    ax.set_ylim(plotrange[1,0], plotrange[1,1])
    ax.set_zlim(plotrange[2,0], plotrange[2,1])
    # labels[insidePts] = labels[outsidePts].min()
    # scL = rescale(labels[all_pts],labels[all_pts])
    # #filted= np.random.choice(ret.shape[0], 5000, replace=False)

    # cmapF = mpl.cm.get_cmap(cmap)
    # rgba_colors = cmapF(scL)
    # rgba_colors[:, 3] = (1- np.power(scL,.1))*.7+.3
    im = ax.scatter(samples[all_pts,0], samples[all_pts,1], samples[all_pts,2], c=rgba_colors, s=ptsSize, cmap=cmap)#, s=rgba_colors[:,3]*10)#, mpl.rcParams['lines.markersize'] ** 2/5.)
    if axis==False:
        ax.set_axis_off()
    #addColorbar2Plot(fig, ax, im, 'bottom') 
    if title is None:
        title = "min:%.3f max:%.3f"%(labels.min(), labels.max())
    fig.suptitle(title)
    #plt.close(fig)
    return fig, ax 

def field3DPlot(outDir, figName, planes, cmap=sdf_cmap,sdfPlot=False, video=False):
    figDir  = os.path.join(outDir, figName)
    summaryPath = os.path.join(outDir, figName+'.png')
    util.mkdirs(figDir)
    norm = None
    if sdfPlot ==True:
        norm='sdf'
    figs = [densityPlot(samples=None,values=planes[i], cmap=cmap, norm=norm, bounds=np.array([planes.min(),planes.max()]))[0] \
                for i in range(planes.shape[0])]
    grid = figGrid(figs)
    saveImg(summaryPath, grid)
    if video:
        vidPath = os.path.join(outDir, figName+'.mp4')
        saveFigs(targetDir=figDir, baseName='', figs=figs)
        util.imgs2video(targetDir=outDir, folderName=figName)
    #plt.close()
    return grid

def plotPCs(pointclouds, cmap='rainbow', plotrange=np.array([[0.,1.],[0.,1.],[0.,1.]]), resolution=(400.,400.), fig=None, ax=None):
    fig, ax = newPlot3D(resolution=resolution, projection='3d',fig=fig, ax=ax)
    for pointcloud in pointclouds:
        all_pts = util.subsampleBoolArray(np.ones(pointcloud.shape[0],dtype=bool), 10000)#./pointclouds.shape[0])
        im = ax.scatter(pointcloud[all_pts,0], pointcloud[all_pts,1], pointcloud[all_pts,2])#, s=rgba_colors[:,3]*10)#, mpl.rcParams['lines.markersize'] ** 2/5.)
    return fig, ax
import matplotlib.patches as patches

def OctreePlot2D(tree, dim, depth, **kwargs):
    assert dim==2
    boxcenter, boxlen, tdepth = ptutil.ths2nps(ptutil.tree2bboxes(torch.from_numpy(tree), dim=dim, depth=depth))    
    dflt_kwargs=dict(plotrange=np.array([[-1.,1.],[-1.,1.]])*1.2)
    dflt_kwargs.update(kwargs)
    maxdep = tdepth.max()
    fig, ax = newPlot( **dflt_kwargs )
    for i in range(len(tdepth)):
        dep=tdepth[i]
        length = boxlen[i]
        corner = (boxcenter[i]-boxlen[i])        
        lw=1+.5*np.exp(-dep)
        rect = patches.Rectangle(corner, 2*length, 2*length, linewidth=lw, edgecolor=plt.cm.plasma(dep/maxdep), facecolor='none')
        ax.add_patch(rect)
    return fig, ax

def seabornScatterPlot(samples, labels):
    import pandas as pd
    import seaborn as sns
    df = pd.DataFrame({'x':samples[:,0], 'y':samples[:,1], 'label':labels})
    g = sns.scatterplot(x="x", y="y", hue="label", s=16, data=df,legend='full', palette="Paired")
    g.legend(loc='center right', bbox_to_anchor=(1.35, 0.5), ncol=1)
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.gca().set_aspect('equal', adjustable='box')


def depthmapPlot(depthmap, cmap="inferno", max_clip=1e10, \
                equalize=True, plot_hist=False, transparent=False):
    """ in order to show more details in depth map, normalize the histogram first
    """
    img = depthmap - depthmap[depthmap!=0].min()
    img = np.clip(depthmap, 0., max_clip)
    img = img / img.max()
    valid = (depthmap!=0)
    #img = img**beta
    if equalize==True:
        nimg = nputil.histogram_equalization( img[valid].reshape(-1) )[0]
        img[valid] = nimg
        #mg = img*.8
    if plot_hist == True:
        plt.hist(img[valid].reshape(-1))
    img = plt.get_cmap(cmap)( 1-img )

    img[depthmap==0] = 0
    if transparent==False:
        img[...,-1] = 1.
    return img

def imgs2video(targetDir, folderName, frameRate=6):
    ''' Making video from a sequence of images
    
        Make a video from images with index, e.g. 1.png 2.png 3.png ...
        the images should be in targetDir/folderName/ 
        the output will be targetDir/folderName.mp4 .
        Args:
            targetDir: the output directory
            folderName: a folder in targetDir which contains the images
        Returns:
            stdout: stdout
            stderr: stderr
            returncode: exitcode
    '''
    imgs_dir = os.path.join(targetDir, folderName)
    command = 'ffmpeg -framerate {2} -f image2 -i {0} -c:v libx264 -pix_fmt yuv420p -r 25 {1} -y'.format(  \
            os.path.join(imgs_dir,'%d.png'),                                                \
            os.path.join(targetDir, '%s.mp4'%folderName),                                   \
            frameRate
            )
    print('Executing command: ', command)
    _, stderr, returncode = runCommand(command)
    if returncode!=0:
        print("ERROR happened during making visRecon video:\n error code:%d"%returncode, stderr)

def imgs2video2(imgs_dir, out_path, frameRate=6):
    ''' Making video from a sequence of images
    
        Make a video from images with index, e.g. 1.png 2.png 3.png ...
        the images should be in imgs_dir
        the output will be at out_path
        Args:
            targetDir: the output directory
            folderName: a folder in targetDir which contains the images
        Returns:
            stdout: stdout
            stderr: stderr
            returncode: exitcode
    '''
    command='ffmpeg -framerate {framerate} -f image2 -i {imgs} -c:v libx264 -pix_fmt yuv420p -r 25 {outpath} -y'
    command = command.format(outpath=out_path,imgs=os.path.join(imgs_dir,'%d.png'),framerate=frameRate)
    print('Executing command: ', command)
    _, stderr, returncode = runCommand(command)
    if returncode!=0:
        print("ERROR happened during making video:\n error code:%d"%returncode, stderr)

def imgarray2video(targetPath, img_list, frameRate=6):
    temp_dir = os.path.expanduser('~/.temp_imgarray2video/')
    print(os.path.realpath)
    if os.path.exists(temp_dir):
        raise OSError('Temp folder already exists!')
    else:
        try:
            saveImgs(targetDir=temp_dir, baseName='', imgs=img_list)
            imgs2video2(temp_dir, targetPath, frameRate=frameRate)
        finally:
            os.system('rm -r %s'%temp_dir)


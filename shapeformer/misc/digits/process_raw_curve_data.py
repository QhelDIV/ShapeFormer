import sys
sys.path.append('../')
import h5py
import numpy as np
import util.util as util
import os

def process(path, outpath):
    # raw dataset contains:
    # ['char', 'edge', 'fonts', 'lengths', 'vertex']
    dataDict = util.readh5(path)
    num_shape = dataDict['vertex'].shape[0]
    print(dataDict['vertex'].shape,dataDict['lengths'], num_shape)

    vert = np.array([dataDict['vertex'][i][:dataDict['lengths'][0,i]] for i in range(num_shape)])
    # convert edge starting index 1 to 0
    edge = np.array([dataDict['edge'][i][:dataDict['lengths'][0,i]]-1 for i in range(num_shape)])

    # scale each digit seperately
    scaled = np.array([(vt-vt.mean(axis=0))/np.abs(vt-vt.mean(axis=0)).max() for vt in vert])

    dataDict['vertex']=scaled
    dataDict['edge']=edge

    imggrid = visual(dataDict, os.path.join(os.path.dirname(outpath), 'digits.png'))
    util.writeh5(outpath, dataDict)
def visual(dataDict, outpath):
    import matplotlib.pyplot as plt
    import vis
    plots = []
    for i in range(20):
        asx = dataDict['vertex'][i*10:(i+1)*10]
        for i in range(10):
            plt.scatter(asx[i][:,0], asx[i][:,1])
        plt.xlim(np.array([-1,1]))
        plt.ylim(np.array([-1,1]))
        fig = plt.gcf()
        plots.append( vis.fig2img(fig))
        plt.clf()
    plots=vis.imageGrid(plots)
    vis.saveImg( outpath, plots)
    #plt.imshow(figgrid)
    return plots

if __name__=='__main__':
    process(path   = '/smartscan/nnrecon/datasets/digits/digits_raw.h5', \
            outpath= '/smartscan/nnrecon/datasets/digits/digits.h5')
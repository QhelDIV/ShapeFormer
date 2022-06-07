import sys
sys.path.append('../')
import numpy as np
import h5py
import igl
from util import util
import multiprocessing
infinity = 100000000.
observe_num = 200
num_rays = 64
canvasSize = 100

def save_images(images):
    for i in range(images.shape[0]):
        image = Image.fromarray(images[i]*255).convert("L")
        image.save('images/digit6/%d.png'%i)
def normalize(x):
    return x/np.sqrt((x*x).sum())
def length(x):
    return np.sqrt((x*x).sum())
def point2lineDistance(q, p1, p2):
    d = np.linalg.norm(np.cross(p2-p1, p1-q))/np.linalg.norm(p2-p1)
    return d
def pointSegDistance(q, p1, p2):
    line_vec = p2-p1
    pnt_vec = q-p1
    line_len = np.linalg.norm(line_vec)
    line_unitvec = normalize(line_vec)
    pnt_vec_scaled = pnt_vec * 1.0/line_len
    t = np.dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = line_vec * t
    dist = length(nearest - pnt_vec)
    nearest = nearest + p1
    return (dist, nearest)
def rescale(vertex):
    shifted = vertex-vertex.mean(-2)[..., np.newaxis, :]
    scope  = np.abs(shifted).max((-1,-2))[..., np.newaxis, np.newaxis]
    vertex = shifted / scope * 40 + np.array([50.,50.])
    print('checking scope: ' , scope)
    print('checking vertex: ', vertex.max(-2))
    return vertex
class Sampler():
    def __init__(self, vertex, edge):
        #print("vertex num: %d, edge num: %d"%(vertex.shape[0], edge.shape[0]))
        self.scaleFactor = 1000
        self.eps = 0.000001
        self.vertex, self.edge = vertex, edge

    def ifInside(self, sample):
        while True:
            sTheta = np.random.rand()
            sDir = np.array([np.cos(sTheta), np.sin(sTheta)])
            counter = 0
            ifparallel = False
            for ei in range(self.edge.shape[0]):
                p = self.vertex[self.edge[ei,0]]
                r = self.vertex[self.edge[ei,1]] - p
                q = sample
                s = sDir * self.scaleFactor
                if np.abs(r[0]*s[1] - r[1]*s[0]) < self.eps:
                    ifparallel = True
                    break
                hit, t, u, _  = igl.segments_intersect(np.append(p,0), np.append(r,0), np.append(q,0), np.append(s,0))
                if hit == True:
                    counter += 1
            if ifparallel == True:
                print('ray is parallel to an edge, skipping...')
                continue
            if counter % 2 == 0:
                return True
            else:
                return False
    def genSample(self):
        sample = np.random.rand(2)*2.-1.
        inside = self.ifInside(sample)
        return (sample, inside)
    def generate_samples_boundary(self, sampleDensity=1.):
        label =[]
        sample=[]
        eps = 1.
        for ei in range(self.edge.shape[0]):
            p = self.vertex[self.edge[ei,0]]
            q = self.vertex[self.edge[ei,1]]
            segLen = length(q-p)
            segDir = (q-p)/segLen
            sampleN = np.floor(segLen/sampleDensity).astype(int)
            p90 = np.array([-segDir[1],segDir[0]])
            p270= np.array([segDir[1],-segDir[0]])

            for si in range(sampleN):
                base = p + si*segDir*sampleDensity
                sample.append(base + p90  * eps)
                label.append( 0 )
                sample.append(base + p270 * eps)
                label.append( 1 )
        print("samples near boundary generated: sample num:%d"%len(sample))
        return np.array(sample), np.array(label)
    def seqSampleBoundary(self, sampleDensity=1.):
        label =[]
        sample=[]
        eps = 1.
        for ei in range(self.edge.shape[0]):
            p = self.vertex[self.edge[ei,0]]
            q = self.vertex[self.edge[ei,1]]
            segLen = length(q-p)
            segDir = (q-p)/segLen
            sampleN = np.floor(segLen/sampleDensity).astype(int)
            #print(segLen)
            for si in range(sampleN):
                base = p + si*segDir*sampleDensity
                sample.append(base)
                label.append( 0. )
        print("samples on boundary generated: sample num:%d"%len(sample))
        return np.array(sample), np.array(label)

    def computeSDF(self,sample, label=None):
        if label is None:
            # TODO
            pass
        minD = 1000000000.
        for ei in range(self.edge.shape[0]):
            p1 = self.vertex[self.edge[ei,0]]
            p2 = self.vertex[self.edge[ei,1]]
            dist, _ = pointSegDistance(sample, p1, p2)
            minD = min(minD, dist)
        if label == 1:
            minD = - minD
        return minD
    def uniformSampleSDF(self, size=50, shuffle=False, save=False):
        xlist = np.linspace(-1., 1., size)
        ylist = np.linspace(-1., 1., size)
        #xx, yy = np.meshgrid(x, y, sparse=True)
        samples, labels = [], []
        for x in xlist:
            for y in ylist:
                sample = np.array([x,y])
                inside = self.ifInside(sample)
                minD = self.computeSDF(sample, inside)
                samples.append(sample)
                labels.append(minD)
        self.samples, self.labels = np.array(samples), np.array(labels)
        perm   = np.random.permutation(self.samples.shape[0]) if shuffle==True else np.arange(self.samples.shape[0])
        self.samples, self.labels = self.samples[perm], self.labels[perm]
        return self.samples, self.labels

    def generate_samples(self, sNum=80):
        label =[]
        sample=[]
        eps = 1.
        sIn  = []
        sOut = []
        while len(sIn) + len(sOut) < sNum:
            sample, inside = self.genSample()
            if inside == True and len(sIn) < sNum/2:
                sIn.append(sample)
            if inside == False and len(sOut) < sNum/2:
                sOut.append(sample)
        sAll = sOut + sIn
        label = [0]*len(sOut) + [1]*len(sIn)
        print("samples in/out generated: sample num:%d"%len(sAll))
        print("inside samples: ",sIn)
        print("outside samples: ",sOut)
        return np.array(sAll), np.array(label)
    def genSDF(self):
        s1, l1 = self.generate_samples()
        #s2, l2 = generate_samples_boundary(vertex, edge, sampleDensity=.4)
        sample = s1#np.concatenate([s1, s2], axis=-2)
        label  = l1#np.concatenate([l1, l2], axis=-1)
        sdf = []
        for i in range(sample.shape[0]):
            sdf.append(self.computeSDF(sample[i],label[i]))
        self.samples, self.labels = sample, np.array(sdf)
        return sample, np.array(sdf)
    def genPartialSequence(self, seqLen=16, shuffle=False):
        samples, labels = self.seqSampleBoundary()
        print(samples.shape, labels.shape)
        length = samples.shape[0] - samples.shape[0] % seqLen
        intervalLen = length//seqLen
        samples = samples[:length]
        labels  = labels[:length]
        print(samples.shape, labels.shape)
        samples = samples.reshape(seqLen, intervalLen, 2)
        labels  = labels.reshape(seqLen, intervalLen)
        if shuffle==True:
            perm = np.random.permutation(samples.shape[0])
            samples = samples[perm]
            labels = labels[perm]
        print(samples.shape, labels.shape)
        return samples, labels

''' LEGACY CODE
    def create_single_digit6_dataset():
        with h5py.File('datasets/curve5.h5','r') as f:
            vertex=np.array(f['vertex'])
            edge=np.array(f['edge'])-1
        print("vertex num: %d, edge num: %d"%(vertex.shape[0], edge.shape[0]))
        shifted = vertex-vertex.mean(-2)[..., np.newaxis, :]
        scope  = np.abs(shifted).max((-1,-2))[..., np.newaxis, np.newaxis]
        vertex = shifted / scope * 40 + np.array([50.,50.])
        print('checking scope: ' , scope)
        print('checking vertex: ', vertex.max(-2))
        #sample, label = generate_samples(vertex, edge)
        sample, label = generate_samples_boundary(vertex, edge, sampleDensity=.5)
        #sample = np.concatenate([sample, bsample], axis=-2)
        #label  = np.concatenate([label, blabel], axis=-1)
        print("total samples generated: sample num:%d"%sample.shape[0])

        #sample = np.random.rand(1000,2)*50 + 50
        #label = ((sample-50.)**2).sum(-1) <= 25*25
        with h5py.File('datasets/curve5_run.h5','w') as f:
            f['sample']     = sample
            f['label']      = label
    def create_single_digit6_SDF_dataset():
        with h5py.File('datasets/curve5.h5','r') as f:
            vertex=np.array(f['vertex'])
            edge=np.array(f['edge'])-1
        print("vertex num: %d, edge num: %d"%(vertex.shape[0], edge.shape[0]))
        shifted = vertex-vertex.mean(-2)[..., np.newaxis, :]
        scope  = np.abs(shifted).max((-1,-2))[..., np.newaxis, np.newaxis]
        vertex = shifted / scope * 40 + np.array([50.,50.])
        print('checking scope: ' , scope)
        print('checking vertex: ', vertex.max(-2))
        sampler = Sampler(vertex, edge)
        #sample, label = generate_samples(vertex, edge)
        #sample, label = sampler.genSDF()
        sample, label = sampler.uniformSampleSDF()
        #sample = np.concatenate([sample, bsample], axis=-2)
        #label  = np.concatenate([label, blabel], axis=-1)

        #sample = np.random.rand(2500,2)*50 + 50
        #label = sample[:,0] #(sample[:,0]-50)**2 + (sample[:,1]-50)**2 - 30*30
        print("total samples generated: sample num:%d"%sample.shape[0])
        with h5py.File('datasets/curve5_partial.h5','w') as f:
            f['sample']     = sample
            f['label']      = -label
    def create_multi_digit6_SDF_dataset():
        with h5py.File('datasets/curve5_multi_test.h5','r') as f:
            vertices=np.array(f['vertex'])
            edges=np.array(f['edge'])-1
            lengths=np.array(f['lengths'])
        samples, labels, shapeIds = [], [], []
        for i in range(vertices.shape[0]):
            vertex = vertices[i, 0:lengths[0,i]]
            edge   = edges[i, 0:lengths[1,i]]
            print("vertex num: %d, edge num: %d"%(vertex.shape[0], edge.shape[0]))
            shifted = vertex-vertex.mean(-2)[..., np.newaxis, :]
            scope  = np.abs(shifted).max((-1,-2))[..., np.newaxis, np.newaxis]
            vertex = shifted / scope * 40 + np.array([50.,50.])
            print('checking scope: ' , scope)
            print('checking vertex: ', vertex.max(-2))
            sampler = Sampler(vertex, edge)
            #sample, label = generate_samples(vertex, edge)
            #sample, label = sampler.genSDF()
            sample, label = sampler.uniformSampleSDF()
            #sample = np.concatenate([sample, bsample], axis=-2)
            #label  = np.concatenate([label, blabel], axis=-1)

            #sample = np.random.rand(2500,2)*50 + 50
            #label = sample[:,0] #(sample[:,0]-50)**2 + (sample[:,1]-50)**2 - 30*30
            samples.append( sample )
            labels.append( label )
            shapeIds.append( np.ones(sample.shape[0]).astype(int) * i )
        samples, labels, shapeIds = np.concatenate(samples,axis=0), np.concatenate(labels,axis=0), np.concatenate(shapeIds,axis=0)
        print("sample/label/sid shapes:%s %s %s"%(samples.shape, labels.shape, shapeIds.shape))
        with h5py.File('datasets/multi5_test.h5','w') as f:
            f['sample']     = samples
            f['label']      = -labels
            f['shapeId']    = shapeIds
            f['perm']       = np.random.permutation(samples.shape[0])
    def create_mutlti_partial_SDF():
        ind = 10
        with h5py.File('datasets/multi5.h5','r') as f:
            samples, labels, shapeIds = np.array(f['sample']), np.array(f['label']), np.array(f['shapeId'])
            samples = samples[shapeIds==ind]
            labels = labels[shapeIds==ind]
        pick = (samples[:,0]>.6)&(samples[:,1]<.4)&(labels[:]<.01)&(labels[:]>-.01)
        length = pick.sum()
        print(length)
        with h5py.File('datasets/multi5_partial.h5','w') as f:
            f['sample']     = samples[pick]
            f['label']      = labels[pick]
            f['shapeId']    = np.zeros(length,dtype=int)
            f['perm']       = np.random.permutation(length)
        with h5py.File('datasets/multi5_partial_test.h5','w') as f:
            f['sample']     = samples
            f['label']      = labels
            f['shapeId']    = np.zeros(samples.shape[0],dtype=int)
            f['perm']       = np.random.permutation(samples.shape[0])
    def create_all_digit6_dataset():
        with h5py.File('digit6.h5','r+') as f:
            images=np.array(f['images'])
            fonts=np.array(f['fonts'])
            cam_positions = []
            depths = []
            for i in range(images.shape[0]):
                positioni, directioni, depthi = generate_cams(images[0], views=10)
                cam_positions.append( positioni )
                depths.append( depthi )
            f['camera_pos'] = np.array(cam_positions)
            f['depths']     = np.array(depths)

    def temp():
    with h5py.File('datasets/uniform5_100.h5','r') as f:
        tem     = np.array(f['sample'])
        label     = np.array(f['label'])
    with h5py.File('datasets/t_uniform5_100.h5','w') as f:
        f['sample'] = tem
        f['label'] = label
'''

class DigitSDFDataCreator():
    def __init__(self, dsetname, inpath, outdir, partialIndex, rot_degree=None):
        self.dsetname, self.partialIndex, self.rot_degree = dsetname, partialIndex, rot_degree
        rawdata={}
        self.outdir = outdir
        self.inDict = util.readh5(inpath)
        self.vertices, self.edges = self.inDict['vertex'], self.inDict['edge']
        self.shapeNum = self.inDict['vertex'].shape[0]
        # for i in range(self.shapeNum):
        #     vertex = rescale( rawdata['vertex'][i, 0:rawdata['lengths'][0,i]] )
        #     if self.rot_degree is not None:
        #         trans = np.array([0.,0.])
        #         vertex = np.dot(util.get2DRotMat(self.rot_degree), (vertex - trans).T ).T + trans
    def gather(self,i):
        sample, label = Sampler(self.vertices[i], self.edges[i]).uniformSampleSDF( size = int(np.sqrt(self.sampleNum)) )
        shapeId = np.ones(sample.shape[0]).astype(int) * i
        return (sample, label, shapeId)
    def createTrainingData(self, sampleNum):
        inDict, samples, labels, shapeIds, fonts, chars = self.inDict, [], [], [], [], []
        outDict={}
        outDict.update(inDict)
        self.sampleNum = sampleNum
        print(sampleNum, self.shapeNum)
        #return
        #self.gather(0)
        samples, labels, shapeIds = util.parallelMap(self.gather, args=[range(self.shapeNum)], zippedIn=False, zippedOut=False)
        # print(out1,out2,out3)
        # for i in range(self.shapeNum):
        #     sampler = Sampler(self.vertices[i], self.edges[i])
        #     sample, label = sampler.uniformSampleSDF( size = np.sqrt(sampleNum) )
        #     samples.append( sample )
        #     labels.append( label )
        #     shapeIds.append( np.ones(sample.shape[0]).astype(int) * i )
        samples, labels, shapeIds = np.concatenate(samples,axis=0), np.concatenate(labels,axis=0), np.concatenate(shapeIds,axis=0)
        print("sample/label/sid shapes:%s %s %s"%(samples.shape, labels.shape, shapeIds.shape))

        outDict['sample']     = samples
        outDict['label']      = -labels # inside -> negative
        outDict['shapeId']    = shapeIds
        outDict['perm']       = np.random.permutation(samples.shape[0])
        outpath = self.outdir+'/%s.h5'%self.dsetname
        util.writeh5(outpath, outDict)

    # TODO: migrate to new scheme
    def createTestingData(self):
        raise NotImplementedError()
        dataDict={}
        with h5py.File(self.outdir+'/%s.h5'%self.dsetname,'r') as f:
            for key in f.keys():
                dataDict[key] = np.array(f[key])
        valid_ind = (np.abs(dataDict['label'])<2)*(dataDict['shapeId']==0)
        
        dataDict['sample']      = dataDict['sample'][valid_ind]
        dataDict['label']       = dataDict['label'][valid_ind]
        dataDict['shapeId']     = dataDict['shapeId'][valid_ind]
        dataDict['perm']        = dataDict['perm'][valid_ind]
        with h5py.File(self.outdir+'/%s_test.h5'%self.dsetname,'w') as f:
            for key in dataDict.keys():
                f[key] = dataDict[key]

    # TODO: migrate to new scheme
    def createPartialData(self, shapeIndex=0):
        raise NotImplementedError()
        index = shapeIndex
        vertex, edge, rawdata = self.vertices[index], self.edges[index], self.rawdata
        sampler = Sampler(vertex, edge)
        samples, labels = sampler.genPartialSequence(shuffle=True)
        print("sample/label/sid shapes:%s %s"%(samples.shape, labels.shape))
        with h5py.File(self.outdir+'/%s_partial.h5'%self.dsetname,'w') as f:
            f['sample']     = samples
            f['label']      = -labels
            f['shapeId']    = np.zeros_like(labels).astype(int)
            f['font']       = rawdata['fonts'][index]
            f['char']       = rawdata['char'][index]
            f['perm']       = np.random.permutation(samples.shape[0])
            output=dict([(key,np.array(f[key])) for key in f.keys()])
        with h5py.File(self.outdir+'/%s.h5'%self.dsetname,'r') as f:
            test_sample = np.array(f['sample'])
            test_label = np.array(f['label'])
            test_shapeId = np.array(f['shapeId'])
        with h5py.File(self.outdir+'/%s_partial_test.h5'%self.dsetname,'w') as f:
            f['sample']     = test_sample[test_shapeId==index]
            f['label']      = test_label[test_shapeId==index]
            f['shapeId']    = np.zeros(test_sample.shape[0],dtype=int)
            f['font']       = rawdata['fonts'][index]
            f['char']       = rawdata['char'][index]
            f['perm']       = np.random.permutation(test_sample.shape[0])

        return output

import igl
def Plys2HDF5(plyFolder, h5path, plyBaseName='out'):
    raise NotImplementedError()
    files = os.listdir(plyFolder)
    for i in range(len(files)):
        plyname = "%s%d.ply"%(plyBaseName,i)
        plypath = os.path.join(plyFolder, plyname)
        igl.read_


if __name__=='__main__':
    # legacy
    #create_all_digit6_dataset()
    #create_single_digit6_dataset()
    #create_single_digit6_SDF_dataset()
    #create_multi_digit6_SDF_dataset()
    #create_full_SDF_partial()
    #temp()
    #create_full_SDF_dataset()
    inpath = '/smartscan/nnrecon/datasets/digits/digits.h5'
    outdir= '/smartscan/nnrecon/datasets/digits/'
    dsetname = 'digits_sdf'
    dataCreator = DigitSDFDataCreator(dsetname, inpath, outdir, partialIndex=0)#, rot_degree=90.)
    dataCreator.createTrainingData(2500)
    #dataCreator.createTestingData()
    #dataCreator.createPartialData(shapeIndex=195)

"""This module contains simple helper functions """
from __future__ import print_function
import re
import os
import sys
import yaml
import shutil
import importlib
import subprocess


import h5py
import numpy as np
#import torch
from PIL import Image
from scipy.interpolate import RegularGridInterpolator

from collections.abc import Iterable
import time
from tqdm import tqdm

EPS = 0.000000001
## python utils
def dictAdd(dict1, dict2):
    for key in dict2:
        if key in dict1:
            dict1[key]+= dict2[key]
        else:
            dict1[key] = dict2[key]
    return dict1
def dictAppend(dict1, dict2):
    for key in dict2:
        if key in dict1:
            dict1[key].append(dict2[key])
        else:
            dict1[key] = [dict2[key]]
    return dict1
def dictApply(func, dic):
    for key in dic:
        dic[key] = func(dic[key])
def dictMean(dicts):
    keys = dicts[0].keys()
    accum = {k: np.stack([x[k] for x in dicts if k in x]).mean() for k in keys}
    return accum
import collections.abc
def dictUpdate(d1: dict, d2: dict, recursive=True, depth=1):
    """Updates dictionary recursively.

    Args:
        d1 (dict): first dictionary to be updated
        d1 (dict): second dictionary which entries should be used

    Returns:
        collections.abc.Mapping: Updated dict
    """
    for k, d2_v in d2.items():
        d1_v = d1.get(k,None)
        typematch = type(d1_v) is type(d2_v)
        recur = isinstance(d2_v, collections.abc.Mapping) and recursive==True
        if typematch and recur:
            d1[k] = dictUpdate(d1_v, d2_v, depth=depth+1)
        else:
            d1[k] = d2_v
    return d1
def prefixDictKey(dic, prefix=''):
    return dict([(prefix+key, dic[key]) for key in dic])
pj=os.path.join
def strNumericalKey(s):
    if s:
        try:
            c = re.findall('\d+', s)[0]
        except:
            c = -1
        return int(c)

def recursiveMap(container, func):
    # TODO
    raise NotImplementedError()
    # valid_container_type = [dict, tuple, list]
    # for item in container:
    #     if item in valid_container_type:
    #         yield type(item)(recursive_map(item, func))
    #     else:
    #         yield func(item)    
# time
class Timer():
    def __init__(self):
        self.time_stamps=[]
        self.update(print_time=False)
    def update(self, print_time=True):
        self.time_stamps.append(time.time())
        if print_time:
            print(self.time_stamps[-1]-self.time_stamps[-2])
# misc
def listdir(directory, return_path=True):
    """List dir and sort by numericals (if applicable)

    Args:
        directory (str): string repr of directory
        return_path (bool, optional): Whether return full file path instead of just file names. Defaults to True.

    Returns:
        list of str: see 'return_path'
    """
    filenames = os.listdir(directory)
    filenames.sort(key=strNumericalKey)
    if return_path==True:
        paths = [os.path.join(directory, filename) for filename in filenames]
        return filenames, paths
    else:
        return filenames
def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)
def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
def filename(path, suffix=False):
    name = path.split("/")[-1]
    if suffix==False:
        name = ".".join(name.split(".")[:-1])
    return name
def load_module_object(module_path, object_name):
    modulelib = importlib.import_module(module_path)
    exists = False
    for name, cls in modulelib.__dict__.items():
        if name == object_name:
            obj = cls
            if exists == True:
                print(f'WARNING: multiple objects named {object_name} have been found in {module_path}')
            exists = True
    if exists==False:
        raise NameError(f'Object {object_name} not found in {module_path}')
    return obj
def load_object(object_path):
    splited = object_path.split('.')
    module_path = '.'.join(splited[:-1])
    object_name = splited[-1]
    return load_module_object(module_path, object_name)
def instantiate_from_opt(opt):
    if not "class" in opt or opt["class"] is None:
        return None
    return load_object(opt["class"])(**opt.get("kwargs", dict()))
def get_filename(path):
    fname = os.path.basename(path)
    fname = ".".join(fname.split(".")[:-1])
    return fname
def runCommand(command):
    import bashlex
    cmd = list(bashlex.split(command))
    p = subprocess.run(cmd, capture_output=True) # capture_output=True -> capture stdout&stderr
    stdout = p.stdout.decode('utf-8')
    stderr = p.stderr.decode('utf-8')
    return stdout, stderr, p.returncode
def array2batches(array, chunkSize=4):
    return [array[i*chunkSize:(i+1)*chunkSize] for i in range( (len(array)+chunkSize-1)//chunkSize )]
def progbar(array, total=None):
    if total is None:
        if isinstance(array, enumerate):
            array = list(array)
        total = len(array)
    return tqdm(array,total=total, ascii=True)
def parallelMap(func, args, batchFunc=None, zippedIn=True, zippedOut=False, cores=-1, quiet=False):
    from pathos.multiprocessing import ProcessingPool
    """Parallel map using multiprocessing library Pathos

    Args:
        stderr (function): func
        args (arguments): [arg1s, arg2s ,..., argns](zippedIn==False) or [[arg1,arg2,...,argn], ...](zippedIn=True)
        batchFunc (func, optional): TODO. Defaults to None.
        zippedIn (bool, optional): See [args]. Defaults to True.
        zippedOut (bool, optional): See [Returns]. Defaults to False.
        cores (int, optional): How many processes. Defaults to -1.
        quiet (bool, optional): if do not print anything. Defaults to False.

    Returns:
        tuples: [out1s, out2s,..., outns](zippedOut==False) or [[out1,out2,...,outn], ...](zippedOut==True)
    """    
    if batchFunc is None:
        batchFunc = lambda x:x
    if zippedIn==True:
        args = list(map(list, zip(*args))) # transpose
    if cores==-1:
        cores = os.cpu_count()
    pool = ProcessingPool(nodes=cores)
    batchIdx = list(range(len(args[0])))
    batches = array2batches(batchIdx, cores)
    out = []
    iterations = enumerate(batches) if quiet==True else progbar(enumerate(batches))
    for i,batch in iterations:
        batch_args = [[arg[i] for i in batch] for arg in args]
        out.extend( pool.map(func, *batch_args) )
    if zippedOut == False:
        if type(out[0]) is not tuple:
            out=[(item,) for item in out]
        out = list(map(list, zip(*out)))
    return out
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
    command = 'ffmpeg -framerate {2} -f image2 -i {0} -c:v libx264 -crf 20 -pix_fmt yuv420p -r 25 {1} -y'.format(  \
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
    command=("ffmpeg -framerate {framerate} -f image2 -i {imgs} -c:v libx264 -crf 20 -pix_fmt yuv420p "
                    "-r 25 {outpath} -y")
    '''
        -framerate: input framerate. total_time = total_frams/framerate
        -r: output framerate 
    '''
    command = command.format(outpath=out_path,imgs=os.path.join(imgs_dir,'%d.png'),framerate=frameRate)
    print('Executing command: ', command)
    _, stderr, returncode = runCommand(command)
    if returncode!=0:
        print("ERROR happened during making video:\n error code:%d"%returncode, stderr)
def imgarray2video(targetPath, img_list, duration=3.,keep_imgs=False):
    from xgutils.vis import visutil
    temp_dir = os.path.expanduser('~/.temp_imgarray2video/')
    print(os.path.realpath)
    if os.path.exists(temp_dir):
        raise OSError('Temp folder already exists!')
    else:
        if keep_imgs == True:
            img_folder = ''.join(targetPath.split('.')[:-1])
            visutil.saveImgs(targetDir=img_folder, baseName='', imgs=img_list)
        try:
            # convert RGBA to RGB via alpha blending
            img_list = np.array(img_list)
            if img_list.max() > 1.01:
                img_list = img_list/256.
            if len(img_list.shape)==4:
                if img_list.shape[-1]==4: # RGBA image
                    img_list = (1.-img_list[:,:,:,3:])*np.ones_like(img_list[:,:,:,:3]) + \
                            (0.+img_list[:,:,:,3:])*img_list[:,:,:,:3]
                    #img_list = [ img[:,:,3:]*np.ones_like(img[:,:,:3]) + (1.-img[:,:,3:])*img[:,:,:3] for img in img_list ]
            print(img_list.shape, img_list.min(), img_list.max())
            visutil.saveImgs(targetDir=temp_dir, baseName='', imgs=img_list)
            frameRate = len(img_list)/duration
            imgs2video2(temp_dir, targetPath, frameRate=frameRate)
        finally:
            os.system('rm -r %s'%temp_dir)
def sh2option(filepath, parser, quiet=False):
    with open(filepath, 'r') as file:
        data = file.read().replace('\n', ' ')
    args = [string.lstrip().rstrip() for string in bashlex.split(data)][1:]
    args = [string for string in args if string != '']
    previous_argv = sys.argv 
    sys.argv = args
    opt = parser.parse(quiet=quiet)
    sys.argv = previous_argv
    return opt
def isIterable(object):
    return isinstance(object, Iterable)
def make_funcdict(d=None, **kwargs):
    def funcdict(d=None, **kwargs):
        if d is not None:
            funcdict.__dict__.update(d)
        funcdict.__dict__.update(kwargs)
        return funcdict.__dict__
    funcdict(d, **kwargs)
    return funcdict
def pickleCopy(obj):
    import io
    import pickle
    buf = io.BytesIO()
    pickle.dump(obj, buf)
    buf.seek(0)
    obj2 = pickle.load(buf) 
    return obj2
def makeArchive(folder, outpath, format='zip'):
    folder = os.path.abspath(folder)
    folder_name = os.path.basename(folder)
    parent = os.path.dirname(folder)
    #print(folder, folder_name, parent)
    out_file = outpath + '.%s'%format
    if os.path.exists(out_file):
        os.remove(out_file)
    shutil.make_archive(base_name= outpath,format=format, \
                        root_dir = parent, \
                        base_dir = folder_name)
def yamldump(opt, target):
    with open(target, 'w') as file:
        documents = yaml.dump(opt, file)
# H5 dataset
class H5DataDict():
    def __init__(self, path, cache_max = 10000000):
        self.path = path
        f = H5File(path)
        self.fkeys = f.fkeys
        self.dict = dict([(key, H5Var(path, key)) for key in self.fkeys])
        self.cache= dict([(key, {}) for key in self.fkeys])
        self.cache_counter=0
        self.cache_max = cache_max
        f.close()
    def keys(self):
        return self.fkeys
    def __getitem__(self,values):
        if type(values) is not tuple:
            if values in self.fkeys:
                return self.dict[values]
            else:
                raise ValueError('%s does not exist'%values)
        if values[0] in self.fkeys:
            if values[1] not in self.cache[values[0]]:
                data = self.dict[values[0]][values[1]]
                if self.cache_counter < self.cache_max:
                    self.cache[values[0]][values[1]] = data
            else:
                data = self.cache[values[0]][values[1]]
            return data
        else:
            raise ValueError('%s does not exist'%values)
class H5Var():
    def __init__(self, path, datasetName):
        self.path, self.dname=path, datasetName
    def __getitem__(self, index):
        f = H5File(self.path)
        if index is None:
            data = f[(self.dname,)]
        else:
            data = f[self.dname, index]
        f.close()
        return data
    def __len__(self):
        f = H5File(self.path)
        leng = f.getLen(self.dname)
        f.close()
        return leng
    @property
    def shape(self):
        return len(self)
    def append(self, array):
        f = H5File(self.path, mode='a')
        if self.dname not in f.f.keys():
            if np.issubdtype(array[0].dtype, np.integer):
                dtype = 'i8' # 'i' means 'i4' which only support number < 2147483648
            elif np.issubdtype(array[0].dtype, np.float):
                dtype = 'f8'
            else:
                raise ValueError('Unsupported dtype %s'%array.dtype)
            f.f.create_dataset(self.dname, (0,), dtype=dtype, maxshape=(None,), chunks=(102400,))
            f.f.create_dataset('%s_serial_dataBias'%self.dname, (0,), dtype='i', maxshape=(None,), chunks=(102400,))
            f.f.create_dataset('%s_serial_shape'%self.dname, (0,), dtype='i', maxshape=(None,), chunks=(102400,))
            f.f.create_dataset('%s_serial_shapeBias'%self.dname, (0,), dtype='i', maxshape=(None,), chunks=(102400,))
        if "%s_serial_dataBias"%self.dname not in f.f.keys():
            raise('Appending for Non-serialized form is not implemented')
        #f.append(self.dname, array)
        serialData, dataBias, serialShape, shapeBias = serializeArray(array)
        key = self.dname
        dataTuple = [serialData, serialShape]
        for i,key in enumerate([self.dname, '%s_serial_shape'%self.dname]):
            oshape = f.f[key].shape[0]
            f.f[key].resize((oshape+dataTuple[i].shape[0],))
            f.f[key][oshape:oshape+dataTuple[i].shape[0]] = (dataTuple[i] if 'Bias' not in key else dataTuple[i]+f.f[key][-1])
        dataTuple = [dataBias, shapeBias]
        for i,key in enumerate(['%s_serial_dataBias'%self.dname, '%s_serial_shapeBias'%self.dname]):
            oshape = f.f[key].shape[0]
            if oshape ==0:
                f.f[key].resize((dataTuple[i].shape[0],))
                f.f[key][:] = dataTuple[i]
            else:
                tshape = oshape+dataTuple[i].shape[0]-1
                f.f[key].resize((oshape+dataTuple[i].shape[0]-1,))
                f.f[key][oshape:tshape] = dataTuple[i][1:]+f.f[key][oshape-1]
def get_h5row(path, datasetName, index):
    f = H5File(path)
    data = f[datasetName, index]
    f.close()
    return data
class H5File():
    def __init__(self, path, mode='r'):
        self.f = h5py.File(path, mode)
        self.fkeys = list(self.f.keys())
    def keys(self):
        return self.fkeys
    def get_serial_data(self, key, index):
        f = self.f
        serial_data = f[key]
        shapeBias = f['%s_serial_shapeBias'%key]
        dataBias = f['%s_serial_dataBias'%key]
        serial_shape = f['%s_serial_shape'%key]
        shape = np.array( serial_shape[shapeBias[index]:shapeBias[index+1]] )

        data = np.array(serial_data[ dataBias[index]:dataBias[index+1]]).reshape(shape)
        return data
    def __getitem__(self, value):
        f, fkeys = self.f, self.fkeys
        key = value[0]
        if "%s_serial_dataBias"%key in fkeys:
            if len(value)==1:
                serialData, dataBias, serialShape, shapeBias = np.array(f[key]), np.array(f['%s_serial_dataBias'%key]), np.array(f['%s_serial_shape'%key]), np.array(f['%s_serial_shapeBias'%key])
                item = serialized2array(serialData, dataBias, serialShape, shapeBias)
            else:
                if isIterable(value[1]):
                    item = np.array([self.get_serial_data(key, ind) for ind in value[1]])
                else:
                    item = self.get_serial_data(key,value[1])
        elif "%s_shapes"%key in fkeys:
            item = padded2array(f[key][value[1]], f["%s_shapes"%key])
        else:
            if len(value)==1:
                item = np.array(f[key])
            else:
                if isIterable(value[1]):
                    ind  = np.array(value[1])
                    uind, inverse = np.unique(ind, return_inverse=True)
                    sindi= np.argsort(uind)
                    sind = uind[sindi]
                    item = np.array(f[key][ list(sind) ])
                    item = item[sindi]
                    item = item[inverse]
                else:
                    item = np.array(f[key][value[1]])
        #print(type(item),item.shape)
        return item
    def getLen(self, key):
        if "%s_serial_dataBias"%key in self.fkeys:
            leng = self.f["%s_serial_dataBias"%key].shape[0] - 1
        else:
            leng = self.f[key].shape[0]
        return leng
    def append(self, dname, array):
        pass

    def close(self):
        self.f.close()
def readh5(path):
    dataDict={}
    with h5py.File(path,'r') as f:
        fkeys = f.keys()
        for key in fkeys:
            if "_serial" in key:
                continue
            if "_shapes" in key:
                continue
            # if np.array(f[key]).dtype.type is np.bytes_: # if is string (strings are stored as bytes in h5py)
            #     print(f[key])
            #     xs=np.array(f[key])
            #     print(xs, xs.dtype, xs.dtype.type)
            #     dataDict[key] = np.char.decode(np.array(f[key]), encoding='UTF-8')
            #     continue

            if "%s_serial_dataBias"%key in fkeys:
                serialData, dataBias, serialShape, shapeBias = np.array(f[key]), np.array(f['%s_serial_dataBias'%key]), np.array(f['%s_serial_shape'%key]), np.array(f['%s_serial_shapeBias'%key])
                dataDict[key] = serialized2array(serialData, dataBias, serialShape, shapeBias)
            elif "%s_shapes"%key in fkeys:
                dataDict[key] = padded2array(f[key], f["%s_shapes"%key])
            else:
                dataDict[key] = np.array(f[key])
    return dataDict
def writeh5(path, dataDict, mode='w', compactForm='serial', quiet=False):
    with h5py.File(path, mode) as f:
        fkeys = f.keys()
        for key in dataDict.keys():
            if key in fkeys: # overwrite if dataset exists
                del f[key]
            else:
                if dataDict[key].dtype is np.dtype('O'):
                    if compactForm=='padding':
                        padded, shapes = padArray(dataDict[key])
                        f[key] = padded
                        f['%s_shapes'%key] = shapes
                    elif compactForm=='serial':
                        serialData, dataBias, serialShape, shapeBias = serializeArray(dataDict[key])
                        f[key] = serialData
                        f['%s_serial_dataBias'%key] = dataBias
                        f['%s_serial_shape'%key]    = serialShape
                        f['%s_serial_shapeBias'%key]= shapeBias

                elif dataDict[key].dtype.type is np.str_:
                    f[key] = np.char.encode( dataDict[key], 'UTF-8' )
                else:
                    f[key] = dataDict[key]
    if quiet==False:
        print(path, 'is successfully written.')
    return dataDict
def readply(path, scaleFactor=1/256.):
    from plyfile import PlyData, PlyElement

    try:
        #print(path)
        with open(path, 'rb') as f:
            plydata = PlyData.read(f)
        vert = np.array([plydata['vertex']['x'],plydata['vertex']['y'],plydata['vertex']['z']]).T
        face = np.array([plydata['face']['vertex_index'][i] for i in range(len(plydata['face']['vertex_index']))]).astype(int)
        #vert = np.zeros((2,3))
    except:
        print('read error', path)
        return np.zeros((10,3)), np.zeros((10,3)).astype(int), False
    #return vert, face
    return vert*scaleFactor, face, True
def plys2h5(plyDir, h5path, indices=None, scaleFactor=1/256.):
    plynames, plypaths = listdir(plyDir, return_path=True)
    plypaths = np.array(plypaths)
    if indices is None:
        indices = np.arange(len(plynames))
    print('Total shapes: %d, selected shapes: %d'%(len(plynames), indices.shape[0]))
    verts, faces = [], []
    args = [(plypath,) for plypath in plypaths[indices]]
    #verts, faces = parallelMap(readply, args, zippedOut=True)
    func = lambda path: readply(path, scaleFactor=scaleFactor)
    verts, faces, valids = parallelMap(readply, args, zippedOut=False)
    inv_ind = np.where(valids==False)[0]
    print('inv_ind', inv_ind)
    np.savetxt('inv_ind.txt', inv_ind)
    writeh5(h5path, dataDict={'vert':np.array(verts), 'face':np.array(faces), 'index':np.array(indices)}, compactForm='serial')
    return inv_ind

class Obj():
    # Just an empty class. Used for conveniently assign member variables
    def __init__(self, dataDict={}, **kwargs):
        self.update(dataDict)
        self.update(kwargs)
    def update(self, dataDict):
        self.__dict__.update(dataDict)
        return self


def unit_test():
    a={'a':{'b':3,'c':4}, 'd':45,           'df':0, 'sx':{'1':{'2':{'deep':'learning'}}}}
    b={'a':{'b':45},      'd':{'b':3,'c':4},        'sx':{'1':{'2':{'is':'NB'}}}}       
    checker =  sysutil.dictUpdate(a,b) == {'a': {'b': 45, 'c': 4},
                'd': {'b': 3, 'c': 4},
                'df': 0,
                'sx': {'1': {'2': {'deep': 'learning', 'is': 'NB'}}}}
    assert(checker)
if __name__ == '__main__':
    unit_test()

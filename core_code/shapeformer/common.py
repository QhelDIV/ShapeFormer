import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import time
from xgutils import *
from einops import rearrange, repeat
import numpy as np

# sparse voxel related
def get_mode(array):
    vals, counts = np.unique(array, return_counts=True)
    mode = vals[np.argmax(counts)]
    return mode
def pth_get_mode(array):
    vals, counts = torch.unique(array.reshape(-1), return_counts=True)
    mode = vals[torch.argmax(counts)]
    return mode
def index2sparse(index):
    """Turn indices to sparse tuple list (pos, val), 
    first get the mode number, consider it as "zero", then remove them.

    Args:
        index (np.ndarray): array of indices

    Returns:
        (array, array): pos_ind and val_ind and mode
    """
    index = index.reshape(-1)
    mode = get_mode(index)
    non_empty = (index!=mode)
    pos_ind = non_empty.nonzero()[0]
    val_ind = index[pos_ind]
    return pos_ind, val_ind, mode


def filter_end_token(indices, end_token=8192 ):
    valids = (indices!=end_token)
    indices = indices[ valids ]
    return indices
def filter_end_tokens(indices, end_tokens=(8192, 4096)):
    # indices: (L, tuple_n)
    end_tokens = np.array(end_tokens)[None, ...]
    valids = ( indices!=end_tokens ).all(axis=1)
    indices = indices[ valids, : ]
    return indices
def unpack_sparse_voxel(val_ind, pos_ind, depth, fill_ind=6360, return_flattened=False):
    grid_dim = 2**depth
    vox = np.zeros(grid_dim**3) + fill_ind
    vox[pos_ind] = val_ind
    if return_flattened==False:
        vox = nputil.array2NDCube(vox, N=3)
    return vox
def convonet_to_nnrecon(array, dim=3, flatten=True):
    grid = nputil.array2NDCube(array.reshape(-1), N=3)
    swaped = np.swapaxes(grid, 0, -1)
    if flatten==True:
        return swaped.reshape(-1)
    else:
        return swaped
def sparse_convonet_to_nnrecon(pos_ind, shape):
    inds = np.stack(np.unravel_index(pos_ind, shape),axis=0)
    inds[[0,1,2],:] = inds[[2,1,0],:]
    ind = np.ravel_multi_index(inds, shape)
    return ind
# torch utilities
def unpack_sparse(sparse, max_length=None, end_tokens=(100,200)):
    """(B*L, tuple_n) -> (B, L, tuple_n)

    Args:
        sparse ([type]): [description]
        max_length ([type], optional): [description]. Defaults to None.
        end_tokens (tuple, optional): [description]. Defaults to (100,200).

    Returns:
        [type]: [description]
    """
    batch_ind   = sparse[:,0]
    raveled_ind = sparse[:,1]
    val         = sparse[:,2]
    #raveled = ptutil.ravel_index(sparse[:,1:-1], shape=shape)
    unique_batch_ind, counts = torch.unique_consecutive(batch_ind, return_counts=True)
    # e.g. [4,4,4,  4, 3, 3, 3, 1, 2, 2]
    repeated_counts = counts.repeat_interleave(counts)
    # e.g. [4,7,8,10] (cumsum)
    # [4,4,4,4,7,7,7,8,10,10]
    repeated_cum = torch.cumsum(counts, axis=0).repeat_interleave(counts)
    # repeated_cum-repeated_counts: shift to right [0,0,0,0,3,3,3,8,10,10]
    # repeated_arange: [0,1,2,3,0,1,2,0,0,1]
    arange = torch.arange(len(repeated_cum)).type_as(sparse)
    repeated_arange = arange - (repeated_cum-repeated_counts)
    # sequence length depends on longest batch item
    end_tokens = torch.tensor(end_tokens)[None,None,:].type_as(sparse)
    tuple_n    = end_tokens.shape[-1]
    target = end_tokens + torch.zeros(len(counts), counts.max()+1, tuple_n).type_as(sparse)
    #print(target)
    # [pos, val]
    target[batch_ind, repeated_arange, 0] = raveled_ind
    target[batch_ind, repeated_arange, 1] = val
    if max_length is not None and target.shape[1]>max_length:
        target = target[:, :max_length, :]
        target[:, max_length-1, :] = end_tokens[:,0,:] # make sure the last column is end_tokens
    return target
def pack_sparse(sparse, end_tokens=(100,200)):
    # sparse: (B, L, tuple_n)
    end_tokens = torch.tensor(end_tokens).type_as(sparse)
    end_tokens = end_tokens[None, None, :].expand_as(sparse)
    # (B, L)
    isnot_end  = (sparse!=end_tokens).all(axis=-1)
    sparse_len = isnot_end.sum(axis=-1)

    nz_ind     = isnot_end.nonzero(as_tuple=False)
    batch_ind  = nz_ind[:,0]
    raveled_ind= sparse[..., 0][nz_ind.split(1, dim=-1)].reshape(-1)
    val        = sparse[..., 1][nz_ind.split(1, dim=-1)].reshape(-1)

    packed_sparse = torch.stack((batch_ind, raveled_ind, val), axis=-1)
    return packed_sparse
def pack_unpack_unittest():
    sparse = torch.tensor([[0,4,1], [0,5,2], [1,1,5], [2,3,2], [2,5,1], [3,0,0]])
    unpacked = unpack_sparse(sparse)
    assert (unpacked - unpack_sparse(pack_sparse(unpacked))).sum()==0
#batch_index2sparse = batch_dense2sparse
def batch_dense2sparse(indices, unpack=True, max_length=None, end_tokens=torch.tensor((100,200))):
    # (B, res, res, res)
    shape = indices.shape[1:]
    # Assuming indices is sparse, so its mode is the empty index
    mode  = torch.mode(indices.view(-1))[0] # [0]: val [1]: ind
    # non-zero indices (nonzero_num, len(shape))
    nz_ind = (indices!=mode).nonzero(as_tuple=False)
    # ravel multi-dimensional indices (B,)
    batch_ind   = nz_ind[:,0]
    raveled_ind = ptutil.ravel_index(nz_ind[:,1:], shape=shape)
    val         = indices[nz_ind.split(1, dim=-1)].reshape(-1)
    packed_sparse = torch.stack((batch_ind, raveled_ind, val), axis=-1)
    if unpack==True:
        unpacked = unpack_sparse(packed_sparse, max_length=max_length, end_tokens=end_tokens)
        return unpacked, mode
    else:
        return packed_sparse, mode
def batch_sparse2dense(sparse, empty_ind, dense_res, return_flattened=False, dim=3):
    # sparse: (K, 3) should be packed
    batch_ind   = sparse[:,0]
    raveled_ind = sparse[:,1]
    val         = sparse[:,2]
    unique_batch_ind, counts = torch.unique_consecutive(batch_ind, return_counts=True)
    batch_size = len(unique_batch_ind)
    
    dense    = empty_ind + torch.zeros(batch_size, *((dense_res,)*dim)).type_as(sparse)
    # (K, dim)
    grid_ind = ptutil.unravel_index(raveled_ind, shape=(dense_res,)*dim)
    # fill in the values
    dense[(batch_ind[:,None], *grid_ind.split(1, dim=-1))] = val[:,None]
    if return_flattened==True:
        #vox = ptutil.array2NDCube(vox, N=3)
        return dense.reshape(batch_size,-1)
    return dense
def batch_sparse_dense_unittest():
    testA = torch.zeros(2,2,2,2).long()+1
    testA[0,1,1,1] = 2
    testA[0,1,1,0] = 3
    testA[0,1,0,0] = 4
    testA[1,0,0,0] = 7
    testA[1,0,0,1] = 2
    print(testA)
    packed, mode = batch_dense2sparse(testA, unpack=False)
    print(packed)
    dense = batch_sparse2dense(packed, empty_ind=mode, dense_res=2)
    ds = batch_sparse2dense( batch_dense2sparse(dense,unpack=False)[0], empty_ind=mode, dense_res=2)
    print(testA.shape, dense.shape)
    print((testA-ds).sum())


def point2voxel(points, grid_dim=32, ret_coords=False):
    """Voxelize point cloud, [i][j][k] correspond to x, y, z directly

    Args:
        points (torch.Tensor): [B,num_pts,x_dim]
        grid_dim (int, optional): grid dimension. Defaults to 32.

    Returns:
        torch.Tensor: [B,(grid_dim,)*x_dim]
    """
    if type(points) is np.ndarray:
        points = torch.from_numpy(points).float()
    voxel = torch.zeros(points.shape[0], *((grid_dim,)*points.shape[-1])).type_as(points)
    inds = point2index(points, grid_dim)
    # make all the indices flat to avoid using for loop for batch
    # (B*num_points, x_dim)
    inds_flat = inds.view(-1,points.shape[-1])
    # [1,2,3] becomes [1,1,1,...,2,2,2,...,3,3,3,...]
    binds = torch.repeat_interleave(torch.arange(points.shape[0]).type_as(points).long(), points.shape[1])
    if points.shape[-1]==2:
        voxel[binds, inds_flat[:,0], inds_flat[:,1]] = 1
    if points.shape[-1]==3:
        voxel[binds, inds_flat[:,0], inds_flat[:,1], inds_flat[:,2]] = 1
    if ret_coords==True:
        x_dim = points.shape[-1]
        coords = nputil.makeGrid(bb_min=[-1,]*x_dim, bb_max=[1,]*x_dim, shape=[grid_dim,]*x_dim, indexing="ij")
        coords = torch.from_numpy(coords[None,...])
        return voxel, coords
    else:
        return voxel


# pl util
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def init_trained_model_from_ckpt(config):
    model  = sysutil.load_object(config["class"]).load_from_checkpoint(config["ckpt_path"])
    model = model.eval()
    model.train = disabled_train
    return model


# sampling related
def filter_sampling_logits(logits, top_k, top_p, temperature, filter_value=-float('Inf')):
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    logits = logits / temperature
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits
def sample_logits(logits, num_samples=1, **filter_kwargs):
    # logits: (B, vocab_size)
    # apply softmax to convert to probabilities
    filtered_logits = [filter_sampling_logits(logits[i], **filter_kwargs) for i in range(logits.shape[0])]
    filtered_logits = torch.stack(filtered_logits)
    probs  = F.softmax(filtered_logits, dim=-1)
    #print("probs\n",probs.numpy())
    ix_sampled = torch.multinomial(probs, num_samples=num_samples, replacement=True)
    # (B, num_samples)
    return ix_sampled
def sample_unittest():
    samples = 10
    logits = torch.tensor([[-1,0,1], [1.0001,-2,1], [0,0,1], [1.01,1,1.02]])
    sampled = sample_logits(logits, num_samples=samples, top_k=1, top_p=.9, temperature=1.)
    print("top_k=1, top_p=.9\n", sampled.numpy())
    sampled = sample_logits(logits, num_samples=samples, top_k=3, top_p=.9, temperature=1.)
    print("top_k=3, top_p=.1\n", sampled.numpy())
    sampled = sample_logits(logits, num_samples=samples, top_k=3, top_p=.999, temperature=1.)
    print("top_k=3, top_p=.9\n", sampled.numpy())


import os, math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import time
from xgutils import *
from einops import rearrange, repeat
import numpy as np
import traceback
import igl

from .common import *
class ShapeFormer(pl.LightningModule):
    def __init__(self,  tuple_n=None, block_size=None, end_tokens=None, vocab_sizes=None, extra_vocab_sizes=None, 
                        voxel_res=16, transformer_opt=None, representer_opt=None, optim_opt={}):
        super().__init__()
        self.__dict__.update(locals())
        self.save_hyperparameters()
        self.transformer = sysutil.instantiate_from_opt(opt=transformer_opt)
        self.representer = sysutil.instantiate_from_opt(opt=representer_opt)
        assert "TupleGPT" in transformer_opt["class"]

    def forward(self, stage="train", **kwargs):
        # get indices from representer
        c_indices, z_indices, extra_indices, others = self.representer.get_indices(stage=stage, **kwargs)
        cz_indices = torch.cat((c_indices, z_indices), dim=1) # (B, L_c+L_s, tuple_n)
        B, L_c, L_z = z_indices.shape[0], c_indices.shape[1], z_indices.shape[1]
        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        targets = z_indices
        # make the prediction
        logits = self.transformer(  idx = cz_indices[:,:-1,:], 
                                    extra_idx = extra_indices[:,:-1,:], 
                                    L_cond = L_c,
                                    target_idx = cz_indices[:,1:,:])
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        for i in range(self.tuple_n):
            # (B, L, vocab_sizes[i])
            logits[i]  = logits[i][..., L_c-1:, :]
            #targets[i] = z_indices[i] #targets[i][:, cond_L-1:, :]
        return logits, targets

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out
    
    @torch.no_grad()
    def sample_indices(self, c_indices, z_indices, max_steps, sample=False, best_in_first=False, top_k=100, top_p=.8, temperature=1.0,
               mask_invalid=True, mask_invalid_completion=False, callback=lambda k: None):
        # x may start as shape (B, 0, tuple_n)
        # c: (B, L_c, tuple_n)
        assert not self.transformer.training
        B, L_c, tuple_n = c_indices.shape
        L_z = z_indices.shape[1]
        L = L_c + L_z
        end_tokens = torch.tensor(self.end_tokens).type_as(z_indices)
        block_size = self.transformer.get_block_size()

        sampled  = torch.zeros(B, L + max_steps, tuple_n).type_as(z_indices)
        seq_head, seq_tail = 0, L
        sampled[:, seq_head:seq_tail, :] = torch.cat((c_indices,z_indices),dim=1)
        logits_history = [[] for i in range(tuple_n)]

        for j in sysutil.progbar(range(max_steps)):
            if seq_tail-seq_head >= block_size:
                # crop generated shape if needed
                sampled[ L_c:seq_tail-1 ] = sampled[ L_c+1:seq_tail ]
                seq_tail -= 1
            c_indices, z_indices = sampled[:, :L_c, :].contiguous(), sampled[:, L_c:seq_tail, :].contiguous()
            idx = sampled[:, :seq_tail, :]
            extra_indices = self.representer.get_extra_indices(c_indices, z_indices)
            #print("extra ",extra_indices[0, L_c:seq_tail, 0])
            #print("idx ", idx[0, L_c:seq_tail, 0])
            #print("i", extra_indices)

            sample_gen = self.transformer.sample_next_tuple(idx, extra_idx=extra_indices, L_cond=L_c)
            # (B, vocab_size)
            logits = next(sample_gen)[:, -1, :] # get newest predict's logit
            for i in range(0, tuple_n): # start from 1, self.tuple_n-1 items in total
                # masking invalid logits
                logits = self.representer.sampling_masker(logits, sampled[:,:seq_tail+1,:], extra_indices, L_cond=L_c, step_j = j, tuple_i = i)
                logits_history[i].append( logits.detach().cpu() )
                # new_ind: (B)
                new_ind      = sample_logits(logits, top_k=top_k, top_p=top_p, temperature=temperature, num_samples=1).reshape(-1)
                new_best_ind = sample_logits(logits, top_k=1,     top_p=0.001, temperature=temperature, num_samples=1).reshape(-1)
                if best_in_first==True:
                    new_ind[0] = new_best_ind[0]
                # fill in the newly generated sample
                sampled[:, seq_tail, i] = new_ind
                if i == tuple_n-1:
                    break
                # (B, seq_head:seq_tail+1)
                target_i = sampled[:, seq_head+1:seq_tail+1, i]
                logits = sample_gen.send(target_i)[:, -1, :] # get newest predict's logit
            seq_tail += 1
            # If all batch encountered stop token, then exit
            no_stop_token = (sampled[:, seq_tail-1, :] != end_tokens[None, :]).all(axis=-1)
            if no_stop_token.long().sum()==0:
                break
        # logits_history: [(B, vocab_size),]*tuple_n
        for i in range(0, tuple_n):
            logits_history[i] = rearrange(logits_history[i], "L B vocab_size -> B L vocab_size")
        # cut off conditioning
        x = sampled[:, L_c:seq_tail, :]
        # convert indices to output
        return x, logits_history
    @torch.no_grad()
    def sample(self, **sampling_kwargs):
        x, logits_history = self.sample_indices(**sampling_kwargs)
        # out_x always should be (raveled_pos, val) tuples
        out_x = self.representer.convert_output_indices(x)
        return out_x, x, logits_history
    def shared_step(self, batch, batch_idx, stage="train"):
        logits, targets = self(**batch, stage=stage)
        loss = 0 
        for i in range(len(logits)):
            logi = rearrange(logits[i], "B L vocab_size -> (B L) vocab_size")
            targ = rearrange(targets[..., i], "B L -> (B L)")
            loss = loss + F.cross_entropy(logi, targ)
        loss = loss / len(logits) # average loss cross all tuple elements
        return loss
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, stage="train")
        self.log("train/loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx, stage="val")
        self.log("val/loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss
    def test_step(self, batch, batch_idx=0, stage='test', return_data=False):
        loss = self.shared_step(batch, batch_idx, stage="test")
        self.log("test/loss", loss, prog_bar=True, logger=True, sync_dist=True)
        return loss


    def configure_optimizers(self):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('cond_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.optim_opt["lr"], betas=(0.9, 0.95))
        return optimizer

class VisShapeFormer(plutil.VisCallback):
    def __init__(self,  temperature=1, sample_n=10, top_k=300, top_p=.9, depth=5, decode_res=128, 
                        sample_max_step=512, render_samples=64, end_tokens=None, 
                        mask_invalid=True, mask_invalid_completion=False, force_keep_c_indices = False, sort_prob=True,
                        partial_radius=0.02, camPos=np.array([2,2,2]), resolution=(512,512), **kwargs):
        self.__dict__.update(locals())
        super().__init__(**kwargs)
        self.vis_camera = dict(camPos=np.array(camPos), camLookat=np.array([0.,0.,0.]),\
                camUp=np.array([0,1,0]),camHeight=2,resolution=resolution, samples=render_samples)
        self.all_Xtg = nputil.makeGrid([-1,-1,-1.],[1.,1,1],[self.decode_res,]*3, indexing="ij")

    def compute_batch(self, batch, input_name=""):
        c_indices, z_indices, extra_indices, others = self.pl_module.representer.get_indices(stage="test", **batch)
        empty_index = others["empty_index"]
        batch_size, tuple_n = c_indices.shape[0], self.pl_module.tuple_n
        assert(batch_size==1)
        # sample
        c_indices_expanded = c_indices.expand(self.sample_n, -1, -1)
        z_start_indices = z_indices[:, :0, :] # (B, 0, tuple_n)
        sample_indices, origin_sample_indices, logits_history = self.pl_module.sample( \
                c_indices = c_indices_expanded,
                z_indices = z_start_indices.expand(self.sample_n, -1, -1), 
                max_steps=self.sample_max_step,
                temperature=self.temperature,
                sample=True,
                best_in_first=True,
                top_k = self.top_k,
                top_p = self.top_p,
                mask_invalid = self.mask_invalid,
                mask_invalid_completion=self.mask_invalid_completion)

        computed = dict(batch=batch)
        computed["samples"] = sample_indices
        computed["origin_samples"] = origin_sample_indices
        computed["logits_history"] = logits_history
        computed["c_ind"] = others["origin_c_indices"]
        computed["z_ind"] = others["origin_z_indices"]
        computed["empty_index"] = empty_index
        
        computed = ptutil.ths2nps(computed)
        if self.sort_prob == True:
            log_prob = compute_log_probs(computed["samples"], computed["logits_history"])
            computed["log_prob"] = log_prob
        return computed
        
    def visualize_batch(self, computed, input_name=""):
        computed = ptutil.ths2nps(computed)
        batch, samples, c_ind, z_ind = computed["batch"], computed["samples"], computed["c_ind"], computed["z_ind"]
        logits_history = computed["logits_history"]
        # samples: (B, L, tuple_n)
        #samples = samples[1] # pos samples
        imgs={}
        config = dict(  end_tokens=self.end_tokens,
                        dense_depth=self.depth,
                        dense_fill_ind=computed["empty_index"].item(),
                        decoder=self.pl_module.representer.vqvae_model, 
                        Xtg=self.all_Xtg,
                        quant_ind_max=self.end_tokens[1], 
                        camera_kwargs=self.vis_camera)
        if "Xbd" in batch:
            complete_cloud = fresnelvis.renderMeshCloud(cloud=batch["Xbd"][0], **self.vis_camera)
            imgs.update( {"data_pc_c":complete_cloud} )
            imgs.update( vis_ind(indices=z_ind[0], prefix=f"data_z", **config) )

        partial_cloud  = fresnelvis.renderMeshCloud(cloud=batch["Xct"][0], cloudR=self.partial_radius, **self.vis_camera)
        imgs.update( {"data_pc_p":partial_cloud} )
        imgs.update( vis_ind(indices=c_ind[0], prefix=f"data_c", **config) )

        if self.sort_prob==True:
            if "log_prob" in batch:
                log_prob = batch["log_prob"]
            else:
                log_prob = compute_log_probs(samples, logits_history)
            probs = [logp.sum() for logp in log_prob]
            order = np.argsort(np.array(probs))[::-1]
        else:
            order = np.arange(samples.shape[0])
        for i in range(samples.shape[0]):
            origin_i = i
            i = order[origin_i]
            sample = samples[i]            
            if self.force_keep_c_indices:
                # cat order matters, [2,3,4,1,2,3] -> [1(3) 2(0) 3(1) 4(2)] (X) means the unique number is at position X
                # by putting c_indices first, the returned uniind will come from c_indices
                cated = ptutil.ths2nps( np.concatenate([c_ind[0], samples[i]], axis=0) )
                uni, uniind = np.unique(cated[:,0], return_index=True)
                sample = np.stack([uni, cated[uniind, 1]], axis=1)
            imgs.update( vis_ind(indices=sample, prefix=f"s{i}", **config) )

        for key in list(imgs.keys()):
            if "mesh" in key:
                mesh_dir = os.path.join(self.data_dir, "meshes/")
                sysutil.mkdirs(mesh_dir)
                vert, face = imgs[key]["vert"].copy(), imgs[key]["face"].copy()
                del imgs[key]
                if vert.shape[0]<10:
                    continue
                igl.write_triangle_mesh( os.path.join(mesh_dir, input_name+"_"+key+".ply"), vert, face,  force_ascii=False)
        
        return imgs

def vis_ind(    indices=None, prefix="data", 
                end_tokens=None,
                dense_depth=None,
                dense_fill_ind=None, 
                decoder=None,
                Xtg=None,
                quant_ind_max=None,
                camera_kwargs={}):
    imgs = {}
    try:
        filtered = filter_end_tokens(indices, end_tokens=end_tokens)
        if filtered.shape[0]==0: # if this is empty sequence
            dimg = qimg = visutil.blankImg(camera_kwargs["resolution"])
            vert, face = None, None
        else:
            packed_sparse = torch.zeros(filtered.shape[0],3).long()
            packed_sparse[:,1:] = torch.from_numpy(filtered)
            #print(filtered, indices)
            voxel_vqind = batch_sparse2dense(packed_sparse, dense_fill_ind, 2**dense_depth, return_flattened=False, dim=3)[0].numpy()
            occupancy = decode_sample_indices( decoder = decoder, Xtg=Xtg, voxel_vqind=voxel_vqind )

            vert, face = geoutil.array2mesh(occupancy, dim=3, coords=Xtg, thresh=.5, if_decimate=False)
            dimg = fresnelvis.renderMeshCloud(mesh={'vert':vert, 'face':face}, meshC=fresnelvis.gray_color, **camera_kwargs)

            pos_ind, val_ind = filtered[:,0], filtered[:,1]
            pos_ind = sparse_convonet_to_nnrecon(pos_ind, shape=(2**dense_depth,)*3)
            qimg = vis3d.IndexVoxelPlot( pos_ind, val_ind, val_max=quant_ind_max, depth=dense_depth, camera_kwargs=camera_kwargs)

    except Exception as e:
        traceback.print_exc()
        print(e)
        return imgs
    imgs[prefix+"_decoded"]   = dimg
    imgs[prefix+"_quant_ind"] = qimg
    if vert is not None:
        imgs[prefix+"_mesh"]      = {"vert":vert, "face":face}
    return imgs
    
def decode_sample_indices(decoder, Xtg, voxel_vqind):
    Xtg = torch.from_numpy(Xtg[None,...]).to(decoder.device)
    voxel_vqind = torch.from_numpy(voxel_vqind)[None,...].long().to(decoder.device)
    with torch.no_grad():
        decoded = ptutil.ths2nps(decoder.decode_index(voxel_vqind, Xtg = Xtg))
        logits = decoded["logits"]
        occupancy = nputil.sigmoid(logits)[0,...,0]
    torch.cuda.empty_cache()
    return occupancy

def compute_log_prob(sample, logits_history, si):
    """ sample: (L, tuple_n), logits_history: [(L,vocab_size),]*tuple_n
        output: (L, tuple_n)
    """
    tuple_n = len(logits_history)
    slog_p = np.zeros(sample.shape)
    for ti in range(tuple_n):
        sample_ti = sample[...,ti]
        log_prob = nputil.logsoftmax(logits_history[ti][si], axis=-1)
        slog_p[:,ti] = log_prob[np.arange(sample_ti.shape[0]), sample_ti]
    return slog_p

def compute_log_probs(samples, logits_history):
    """ sample: (S, L, tuple_n), logits_history: [(S, L,vocab_size),]*tuple_n
        output: (S, L, tuple_n)
    """
    sample_n, sample_L, tuple_n = samples.shape
    slog_p = np.zeros(samples.shape)
    for si in range(sample_n):
        for ti in range(tuple_n):
            sample_ti = samples[si, :, ti]
            log_prob = nputil.logsoftmax(logits_history[ti][si], axis=-1)
            slog_p[si, :, ti] = log_prob[np.arange(sample_L), sample_ti]
    return slog_p

import os
import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import time
from xgutils import *


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class NaiveSF(pl.LightningModule):
    def __init__(self, transformer_opt, first_stage_opt,
                 cond_stage_opt=None, permuter_opt=None,
                 optim_opt={}):

        super().__init__()
        self.save_hyperparameters()
        self.optim_opt = optim_opt
        self.first_stage_model = self.init_trained_model_from_ckpt(
            first_stage_opt)
        self.cond_stage_model = self.first_stage_model

        self.transformer = sysutil.instantiate_from_opt(opt=transformer_opt)
        if permuter_opt is None:
            #permuter_opt = {"target": "taming.modules.transformer.permuter.Identity"}
            permuter_opt = {
                "class": "shapeformer.models.autoregressive.transformer.permuter.Identity"}
        self.permuter = sysutil.instantiate_from_opt(opt=permuter_opt)

    def init_trained_model_from_ckpt(self, config):
        model = sysutil.load_object(
            config["class"]).load_from_checkpoint(config["ckpt_path"])
        model = model.eval()
        # freeze model, so that its not in grad flow, neccessary for ddp
        model.requires_grad_(False)
        model.train = disabled_train
        return model

    def forward(self, Xbd, Xct, **kwargs):
        # one step to produce the logits
        _, z_indices, _ = self.encode_to_z(Xbd)
        _, c_indices, _ = self.encode_to_c(Xct)

        cz_indices = torch.cat((c_indices, z_indices), dim=1)

        # target includes all sequence elements (no need to handle first one
        # differently because we are conditioning)
        target = z_indices
        # make the prediction
        # print(cz_indices.shape)
        logits, _ = self.transformer(cz_indices[:, :-1])
        # print(logits.shape)
        # cut off conditioning outputs - output i corresponds to p(z_i | z_{<i}, c)
        logits = logits[:, c_indices.shape[1]-1:]
        #print("cuted", logits.shape)

        return logits, target

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, steps, temperature=1.0, sample=False, deterministic_last=False, top_k=None,
               callback=lambda k: None):
        # x may start as shape (B, 0, )
        x = torch.cat((c, x), dim=1)
        block_size = self.transformer.get_block_size()
        assert not self.transformer.training

        for k in sysutil.progbar(range(steps)):
            # print(k)
            callback(k)
            # make sure model can see conditioning
            assert x.size(1) <= block_size
            # crop context if needed
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
            logits, _ = self.transformer(x_cond)
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely

            _, best_ix = torch.topk(probs, k=1, dim=-1)
            ix_sampled = torch.multinomial(probs, num_samples=1)
            if sample == True:
                ix = ix_sampled
                if deterministic_last == True:
                    ix[-1, ...] = best_ix[-1, ...]
            else:
                ix = best_ix
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)
        # cut off conditioning
        x = x[:, c.shape[1]:]
        return x

    @torch.no_grad()
    def encode_to_z(self, Xtg):
        encoded = self.first_stage_model.encode(Xtg)
        indices = encoded["ind"].view(Xtg.shape[0], -1)
        indices = self.permuter(indices)
        return encoded["quant_feat"], indices, encoded

    @torch.no_grad()
    def encode_to_c(self, Xct):
        # if self.downsample_cond_size > -1:
        #    c = F.interpolate(Xct, size=(self.downsample_cond_size, self.downsample_cond_size))
        encoded = self.cond_stage_model.encode(Xct)
        indices = encoded["ind"].view(Xct.shape[0], -1)
        return encoded["quant_feat"], indices, encoded

    def decode(self, index, zshape, Xtg=None):
        index = self.permuter(index, reverse=True)
        bhw = (index.shape[0], *zshape[2:])
        index = index.reshape(*bhw)
        # (B,C,...)
        quant_z = self.first_stage_model.codebook.get_code(index)
        #quant_z = quant_z.reshape(*bhwc)
        decoded = self.first_stage_model.decode(quant_z, Xtg)
        decoded["image"] = torch.sigmoid(decoded["image"])
        return decoded

    def shared_step(self, batch, batch_idx):
        Xbd, Xct = batch["Xbd"], batch["Xct"]
        logits, target = self(Xbd, Xct)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True,
                 on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True,
                 on_step=True, on_epoch=True, sync_dist=True)
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
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

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

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(
            inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
            % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn]
                        for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn]
                        for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.optim_opt["lr"], betas=(0.9, 0.95))
        return optimizer


class VisShapeFormer(plutil.VisCallback):
    def __init__(self,  temperature=1, sample_n=10, top_k=300, top_p=.9, depth=5, decode_res=128,
                 sample_max_step=512, render_samples=64, end_tokens=None,
                 mask_invalid=True, mask_invalid_completion=False, force_keep_c_indices=False, sort_prob=True,
                 partial_radius=0.02, camPos=np.array([2, 2, 2]), resolution=(512, 512), **kwargs):
        self.__dict__.update(locals())
        super().__init__(**kwargs)
        self.vis_camera = dict(camPos=np.array(camPos), camLookat=np.array([0., 0., 0.]),
                               camUp=np.array([0, 1, 0]), camHeight=2, resolution=resolution, samples=render_samples)
        self.all_Xtg = nputil.makeGrid(
            [-1, -1, -1.], [1., 1, 1], [self.decode_res, ]*3, indexing="ij")

    def compute_batch(self, batch, input_name=""):
        c_indices, z_indices, extra_indices, others = self.pl_module.representer.get_indices(
            stage="test", **batch)
        empty_index = others["empty_index"]
        batch_size, tuple_n = c_indices.shape[0], self.pl_module.tuple_n
        assert(batch_size == 1)
        # sample
        c_indices_expanded = c_indices.expand(self.sample_n, -1, -1)
        z_start_indices = z_indices[:, :0, :]  # (B, 0, tuple_n)
        sample_indices, origin_sample_indices, logits_history = self.pl_module.sample(
            c_indices=c_indices_expanded,
            z_indices=z_start_indices.expand(self.sample_n, -1, -1),
            max_steps=self.sample_max_step,
            temperature=self.temperature,
            sample=True,
            best_in_first=True,
            top_k=self.top_k,
            top_p=self.top_p,
            mask_invalid=self.mask_invalid,
            mask_invalid_completion=self.mask_invalid_completion)

        #print("sampled indices length: ", sample_indices.shape)
        #print("extra", extra_indices)
        #print("sampled indices:", (sample_indices!=4096).sum(axis=1))
        #print("sampled origin indices:", origin_sample_indices)
        computed = dict(batch=batch)
        computed["samples"] = sample_indices
        computed["origin_samples"] = origin_sample_indices
        computed["logits_history"] = logits_history
        computed["c_ind"] = others["origin_c_indices"]
        computed["z_ind"] = others["origin_z_indices"]
        computed["empty_index"] = empty_index

        computed = ptutil.ths2nps(computed)
        if self.sort_prob == True:
            log_prob = compute_log_probs(
                computed["samples"], computed["logits_history"])
            computed["log_prob"] = log_prob
        return computed

    def visualize_batch(self, computed, input_name=""):
        computed = ptutil.ths2nps(computed)
        batch, samples, c_ind, z_ind = computed["batch"], computed["samples"], computed["c_ind"], computed["z_ind"]
        logits_history = computed["logits_history"]
        # samples: (B, L, tuple_n)
        # samples = samples[1] # pos samples
        imgs = {}
        config = dict(end_tokens=self.end_tokens,
                      dense_depth=self.depth,
                      dense_fill_ind=computed["empty_index"].item(),
                      decoder=self.pl_module.representer.vqvae_model,
                      Xtg=self.all_Xtg,
                      quant_ind_max=self.end_tokens[1],
                      camera_kwargs=self.vis_camera)
        if "Xbd" in batch:
            complete_cloud = fresnelvis.renderMeshCloud(
                cloud=batch["Xbd"][0], **self.vis_camera)
            imgs.update({"data_pc_c": complete_cloud})
            imgs.update(vis_ind(indices=z_ind[0], prefix=f"data_z", **config))

        partial_cloud = fresnelvis.renderMeshCloud(
            cloud=batch["Xct"][0], cloudR=self.partial_radius, **self.vis_camera)
        imgs.update({"data_pc_p": partial_cloud})
        imgs.update(vis_ind(indices=c_ind[0], prefix=f"data_c", **config))

        if self.sort_prob == True:
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
                cated = ptutil.ths2nps(np.concatenate(
                    [c_ind[0], samples[i]], axis=0))
                uni, uniind = np.unique(cated[:, 0], return_index=True)
                sample = np.stack([uni, cated[uniind, 1]], axis=1)
            imgs.update(vis_ind(indices=sample, prefix=f"s{i}", **config))

        for key in list(imgs.keys()):
            if "mesh" in key:
                mesh_dir = os.path.join(self.data_dir, "meshes/")
                sysutil.mkdirs(mesh_dir)
                vert, face = imgs[key]["vert"].copy(), imgs[key]["face"].copy()
                del imgs[key]
                if vert.shape[0] < 10:
                    continue
                igl.write_triangle_mesh(os.path.join(
                    mesh_dir, input_name+"_"+key+".ply"), vert, face,  force_ascii=False)

        return imgs


def vis_ind(indices=None, prefix="data",
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
        if filtered.shape[0] == 0:  # if this is empty sequence
            dimg = qimg = visutil.blankImg(camera_kwargs["resolution"])
            vert, face = None, None
        else:
            packed_sparse = torch.zeros(filtered.shape[0], 3).long()
            packed_sparse[:, 1:] = torch.from_numpy(filtered)
            #print(filtered, indices)
            voxel_vqind = batch_sparse2dense(
                packed_sparse, dense_fill_ind, 2**dense_depth, return_flattened=False, dim=3)[0].numpy()
            occupancy = decode_sample_indices(
                decoder=decoder, Xtg=Xtg, voxel_vqind=voxel_vqind)

            vert, face = geoutil.array2mesh(
                occupancy, dim=3, coords=Xtg, thresh=.5, if_decimate=False)
            dimg = fresnelvis.renderMeshCloud(
                mesh={'vert': vert, 'face': face}, meshC=fresnelvis.gray_color, **camera_kwargs)

            pos_ind, val_ind = filtered[:, 0], filtered[:, 1]
            # if "s0" in prefix or "data_c" in prefix:
            #     print(prefix)
            #     print(pos_ind)
            #     print(val_ind)
            pos_ind = sparse_convonet_to_nnrecon(
                pos_ind, shape=(2**dense_depth,)*3)
            qimg = vis3d.IndexVoxelPlot(
                pos_ind, val_ind, val_max=quant_ind_max, depth=dense_depth, camera_kwargs=camera_kwargs)

    except Exception as e:
        #dimg = qimg = np.array(*camera_kwargs["resolution"],3)
        traceback.print_exc()
        print(e)
        return imgs
    imgs[prefix+"_decoded"] = dimg
    imgs[prefix+"_quant_ind"] = qimg
    if vert is not None:
        imgs[prefix+"_mesh"] = {"vert": vert, "face": face}
    return imgs


def decode_sample_indices(decoder, Xtg, voxel_vqind):
    Xtg = torch.from_numpy(Xtg[None, ...]).to(decoder.device)
    voxel_vqind = torch.from_numpy(
        voxel_vqind)[None, ...].long().to(decoder.device)
    with torch.no_grad():
        decoded = ptutil.ths2nps(decoder.decode_index(voxel_vqind, Xtg=Xtg))
        logits = decoded["logits"]
        occupancy = nputil.sigmoid(logits)[0, ..., 0]
    torch.cuda.empty_cache()
    return occupancy


def compute_log_prob(sample, logits_history, si):
    """ sample: (L, tuple_n), logits_history: [(L,vocab_size),]*tuple_n
        output: (L, tuple_n)
    """
    tuple_n = len(logits_history)
    slog_p = np.zeros(sample.shape)
    for ti in range(tuple_n):
        sample_ti = sample[..., ti]
        log_prob = nputil.logsoftmax(logits_history[ti][si], axis=-1)
        slog_p[:, ti] = log_prob[np.arange(sample_ti.shape[0]), sample_ti]
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

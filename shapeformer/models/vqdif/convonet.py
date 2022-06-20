import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter_max

from xgutils import *
from xgutils.vis import npfvis

from einops.layers.torch import Rearrange, Reduce

from shapeformer.models import networks
from shapeformer.models import pointnet
from shapeformer.models.treesformer.common import *

from .quantizer import Quantizer


class ConvONet(pl.LightningModule):
    def __init__(self,  encoder_opt=None, decoder_opt=None, optim_opt=None, ckpt_path=None, opt=None):
        super().__init__()
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.encoder = sysutil.instantiate_from_opt(opt=self.encoder_opt)
        self.decoder = sysutil.instantiate_from_opt(opt=self.decoder_opt)
        self.criterion = nn.BCEWithLogitsLoss()

    def encode(self, Xbd, **kwargs):
        grid_feat, grid_mask = self.encoder(Xbd/2.)  # [-1,1] -> [-.5,.5]
        return grid_feat, grid_mask

    def decode(self, grid_feat, Xtg=None, **kwargs):
        # quant_feat: (res, res, res)
        max_limit = 64**3
        if Xtg.shape[1] > max_limit:
            with torch.no_grad():
                batch_xtg = torch.split(Xtg, max_limit, dim=1)
                b_ytgs = []
                for b_xtg in batch_xtg:
                    b_ytgs.append(self.decoder(b_xtg/2, grid_feat))
                Ytg_logits = torch.cat(b_ytgs, dim=1)
        else:
            Ytg_logits = self.decoder(Xtg/2., grid_feat)  # [-1,1] -> [-.5,.5]
        return dict(logits=Ytg_logits)

    def forward(self, Xbd, Xtg, **kwargs):
        # Xbd: [B, num_pts, x_dim], Xtg: [B, num_probes, x_dim]
        B, num_pts, x_dim = Xbd.shape
        # (B, C, res, res, res)
        grid_feat, grid_mask = self.encode(Xbd)
        logits = self.decode(grid_feat, Xtg=Xtg)["logits"]
        return dict(logits=logits)

    def get_loss(self, batch, batch_idx, stage="train"):
        out = self.forward(**batch)
        losses = self.criterion.get_loss(
            logits=out["logits"], label=batch["Ytg"], quant_diff=out["quant_diff"])
        return losses

    def training_step(self, batch, batch_idx, stage="train"):
        losses = self.get_loss(batch, batch_idx, stage="train")
        for loss_name in losses:
            self.log(f'{stage}/{loss_name}',
                     losses[loss_name], prog_bar=True, sync_dist=True)
        return losses['loss']

    def validation_step(self, batch, batch_idx=0, stage='val', return_data=False):
        losses = self.get_loss(batch, batch_idx, stage="val")
        for loss_name in losses:
            self.log(f'{stage}/{loss_name}',
                     losses[loss_name], prog_bar=False, sync_dist=True)
        self.log('val_loss', losses['loss'], prog_bar=False, sync_dist=True)
        return losses['loss']

    def test_step(self, batch, batch_idx=0, stage='test', return_data=False):
        losses = self.get_loss(batch, batch_idx, stage="test")
        for loss_name in losses:
            self.log(f'{stage}/{loss_name}',
                     losses[loss_name], prog_bar=False, sync_dist=True)
        return losses['loss']

    def configure_optimizers(self):
        oopt = self.optim_opt
        if oopt is None:
            return [], []
        optim = torch.optim.Adam(self.parameters(), lr=self.optim_opt['lr'])

        sched_name = oopt['scheduler']
        if sched_name == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                        step_size=oopt['step_size'],
                                                        gamma=oopt['gamma'],
                                                        verbose=True)
        elif sched_name == 'None':
            return optim
        else:
            raise NotImplementedError(f'Can not use scheduler:{sched_name}')
        return [optim], [scheduler]

    @classmethod
    def unittest(cls, **kwargs):
        B, num_pts, num_probes, x_dim = 2, 1024, 4096, 2
        grid_dim = 64
        Xbd = torch.zeros(B, num_pts, x_dim).cuda()
        Xtg = torch.zeros(B, num_probes, x_dim).cuda()
        model = cls(x_dim, grid_dim=grid_dim).cuda()
        out = model(Xbd, Xtg)
        print('logits: ', out['logits'].shape)
        return model, out


class VisConvONet(plutil.VisCallback):
    def __init__(self, samples=32, quant_grid_depth=4, vocab_size=4096, max_length=512, end_tokens=(4096, 4096), resolution=(512, 512), **kwargs) -> None:
        super().__init__(**kwargs)
        self.__dict__.update(locals())
        #self.visual_indices = list(range(100000))
        self.vis_camera = dict(camPos=np.array([2, 2, 2]), camLookat=np.array([0., 0., 0.]),
                               camUp=np.array([0, 1, 0]), camHeight=2, resolution=resolution, samples=samples)
        self.cloudR = 0.004
        self.all_Xtg = torch.from_numpy(nputil.makeGrid(
            [-1, -1, -1.], [1., 1, 1], [256, ]*3, indexing="ij"))[None, ...]

    def decode_batch(self, quant_ind, batch):
        # quant_ind (1, res, res, res)
        sparse, mode = batch_dense2sparse(
            quant_ind, max_length=self.max_length, end_tokens=self.end_tokens)
        packed_sparse = pack_sparse(sparse, end_tokens=self.end_tokens)
        dense = batch_sparse2dense(
            packed_sparse, empty_ind=mode, dense_res=2**self.quant_grid_depth)

        decoded = self.pl_module.decode_index(
            dense, self.all_Xtg.type_as(batch["Xtg"]))
        if type(decoded) is not dict:
            decoded = dict(logits=decoded)
        decoded = ptutil.ths2nps(decoded)
        ret = {"logits": decoded["logits"], "quant_ind": encoded["quant_ind"],
               "sparse": packed_sparse, "batch": batch}
        return ptutil.ths2nps(ret)

    def compute_batch(self, batch):
        # out = self.pl_module(batch["Xbd"], self.all_Xtg.type_as(batch["Xtg"]))
        # if type(out) is not dict:
        #     out = dict(logits=out)
        # return {"logits":out["logits"], "batch":batch}
        quant_ind, mode, encoded = self.pl_module.quantize_cloud(batch["Xbd"])
        grid_mask = encoded["grid_mask"]
        # quant_ind[quant_ind> =1288]=1391

        #print( torch.stack(torch.unique(quant_ind.reshape(-1), return_counts=True),axis=-1) )
        sparse, mode = batch_dense2sparse(
            quant_ind, max_length=self.max_length, end_tokens=self.end_tokens)
        packed_sparse = pack_sparse(sparse, end_tokens=self.end_tokens)
        dense = batch_sparse2dense(
            packed_sparse, empty_ind=mode, dense_res=2**self.quant_grid_depth)

        decoded = self.pl_module.decode_index(
            dense, self.all_Xtg.type_as(batch["Xbd"]))
        if type(decoded) is not dict:
            decoded = dict(logits=decoded)
        decoded = ptutil.ths2nps(decoded)
        ret = {"logits": decoded["logits"], "quant_ind": encoded.get("quant_ind"),
               "sparse": packed_sparse,     "grid_mask": grid_mask, "batch": batch}
        return ptutil.ths2nps(ret)

    def visualize_batch(self, computed):
        computed = ptutil.ths2nps(computed)
        batch, logits, quant_ind, sparse_quant_ind = computed["batch"], computed[
            "logits"], computed["quant_ind"], computed["sparse"]
        quant_ind = convonet_to_nnrecon(quant_ind)
        occupancy = nputil.sigmoid(logits.reshape(-1))
        Xtg = batch['Xtg'][0] if "Xtg" in batch else None
        all_Xtg = self.all_Xtg.numpy()
        imgs = {}
        if 'Ytg' in batch:
            imgs["gt"] = npfvis.plot_3d_recon(
                Xtg=Xtg, Ytg=batch['Ytg'][0], camera_kwargs=self.vis_camera)
        if "Xbd" in batch:
            imgs["gt_pc"] = fresnelvis.renderMeshCloud(
                cloud=batch["Xbd"][0], cloudR=self.cloudR, **self.vis_camera)
        #imgs["recon"] = npfvis.plot_3d_sample(Xct=batch["Xbd"][0], Yct=None, Xtg=batch['Xtg'][0], Ytg=None, pred_y=occupancy[0], show_images=["pred"])[0]
        #imgs["recon"] = npfvis.plot_3d_sample(Yct=None, Xtg=batch['Xtg'][0], Ytg=None, pred_y=occupancy[0], show_images=["pred"])[0]
        imgs["recon"] = npfvis.plot_3d_recon(
            Xtg=all_Xtg, Ytg=occupancy, camera_kwargs=self.vis_camera)

        pos_ind, val_ind = sparse_quant_ind[:, 1], sparse_quant_ind[:, 2]
        pos_ind = sparse_convonet_to_nnrecon(
            pos_ind, shape=(2**self.quant_grid_depth,)*3)
        imgs["quant_ind"] = vis3d.IndexVoxelPlot(
            pos_ind, val_ind, val_max=self.vocab_size, depth=self.quant_grid_depth, camera_kwargs=self.vis_camera)
        pos_ind = sparse_convonet_to_nnrecon(
            computed["grid_mask"][0].reshape(-1).nonzero()[0], shape=(2**self.quant_grid_depth,)*3)
        imgs["mask_ind"] = vis3d.IndexVoxelPlot(
            pos_ind, pos_ind, val_max=4096, depth=self.quant_grid_depth, camera_kwargs=self.vis_camera)
        return imgs


def codebook_nonzero_count(pl_model):
    codebook = pl_model.quantizer
    #codebook = pl_model.model.codebook
    print(len(codebook.N.nonzero()))
    print(len((codebook.z_avg.abs().sum(axis=-1) > 7.1186e-10).nonzero()))
    w = codebook.embedding.weight
    print(w.shape)
    print(len((w.abs().sum(axis=-1) > 10e-1).nonzero()))

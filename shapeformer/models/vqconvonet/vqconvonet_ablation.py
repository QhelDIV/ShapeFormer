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


class VQConvONet(pl.LightningModule):
    def __init__(self, Xct_as_Xbd=False, encoder_opt=None, decoder_opt=None, quantizer_opt=None, vq_beta=1.,
                 optim_opt=None, ckpt_path=None, opt=None):
        super().__init__()
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.encoder = sysutil.instantiate_from_opt(opt=self.encoder_opt)
        self.decoder = sysutil.instantiate_from_opt(opt=self.decoder_opt)
        self.criterion = VQLoss(beta=vq_beta)

    def encode(self, Xbd, **kwargs):
        grid_feat, grid_mask = self.encoder(Xbd/2.)  # [-1,1] -> [-.5,.5]
        return grid_feat, grid_mask

    def decode(self, grid_feat, Xtg=None, **kwargs):
        # quant_feat: (res, res, res)
        max_limit = 256**3
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

        quant_feat, quant_feat_st, quant_ind, quant_diff = None, None, None, None
        return dict(logits=logits, quant_feat=quant_feat_st, quant_ind=quant_ind, quant_diff=quant_diff, grid_mask=grid_mask)

    def get_loss(self, batch, batch_idx, stage="train"):
        Xbd = batch["Xbd"] if self.Xct_as_Xbd == False else batch["Xct"]
        out = self.forward(Xbd=Xbd, Xtg=batch["Xtg"])
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
                                                        gamma=oopt['gamma'])
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


class VQLoss(nn.Module):
    def __init__(self, beta=1.):
        super().__init__()
        self.beta = beta
        self.bce_logits = nn.BCEWithLogitsLoss()

    def get_loss(self, logits, label, quant_diff=None, **kwargs):
        self.recon_loss = self.bce_logits(logits, label)
        if quant_diff is not None:
            self.diff_loss = quant_diff
            self.loss = self.recon_loss + self.beta * self.diff_loss
            losses = dict(loss=self.loss, recon_loss=self.recon_loss,
                          diff_loss=self.diff_loss)
        else:
            self.loss = self.recon_loss
            losses = dict(loss=self.loss, recon_loss=self.recon_loss)
        return losses


class VisRecon3D(plutil.VisCallback):
    def __init__(self, Xct_as_Xbd=True, decode_res=128, samples=32, resolution=(256, 256), **kwargs) -> None:
        super().__init__(**kwargs)
        self.__dict__.update(locals())

        self.vis_camera = dict(camPos=np.array([2, 2, 2]), camLookat=np.array([0., 0., 0.]),
                               camUp=np.array([0, 1, 0]), camHeight=2, resolution=resolution, samples=samples)
        self.cloudR = 0.01
        self.all_Xtg = torch.from_numpy(nputil.makeGrid(
            [-1, -1, -1.], [1., 1, 1], [decode_res, ]*3, indexing="ij"))[None, ...]

    def compute_batch(self, batch, input_name=""):
        Xbd = batch["Xbd"] if (
            "Xbd" in batch and self.Xct_as_Xbd == False) else batch["Xct"]
        out = self.pl_module(Xbd, self.all_Xtg.type_as(Xbd))
        if type(out) is not dict:
            out = dict(logits=out)
        return {"logits": out["logits"], "batch": batch}

    def visualize_batch(self, computed, input_name=""):
        computed = ptutil.ths2nps(computed)
        batch, logits = computed["batch"], computed["logits"]
        occupancy = nputil.sigmoid(logits)[0, ..., 0]

        imgs = {}
        # if 'Ytg' in batch:
        #    Xtg = batch['Xtg'][0] if "Xtg" in batch else None
        #    imgs["gt"] = npfvis.plot_3d_recon(Xtg=Xtg, Ytg=batch['Ytg'][0], camera_kwargs=self.vis_camera, meshC=fresnelvis.gold_color)
        if "Xbd" in batch:
            imgs["gt_pc"] = fresnelvis.renderMeshCloud(
                cloud=batch["Xbd"][0], cloudR=self.cloudR, **self.vis_camera)
        if "Xct" in batch:
            imgs["xct"] = fresnelvis.renderMeshCloud(
                cloud=batch["Xct"][0], cloudR=self.cloudR, **self.vis_camera)
        imgs["recon"] = npfvis.plot_3d_recon(
            Xtg=self.all_Xtg, Ytg=occupancy, camera_kwargs=self.vis_camera, meshC=fresnelvis.gray_color)
        return imgs

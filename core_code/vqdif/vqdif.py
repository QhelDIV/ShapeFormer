import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.nn as nn
from torch_scatter import scatter_max

from xgutils import *
from xgutils.vis import npfvis

from einops.layers.torch import Rearrange, Reduce

from nnrecon.models import networks
from nnrecon.models import pointnet
from nnrecon.models.shapeformer.common import *

from .quantizer import Quantizer


class VQConvONet(pl.LightningModule):
    def __init__(self, Xct_as_Xbd=False, encoder_opt=None, decoder_opt=None, quantizer_opt=None, vq_beta=1., \
                        optim_opt=None, ckpt_path=None, opt = None):
        super().__init__()
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.encoder  = sysutil.instantiate_from_opt(opt=self.encoder_opt)
        self.decoder  = sysutil.instantiate_from_opt(opt=self.decoder_opt)
        if quantizer_opt is not None:
            self.quantizer = sysutil.instantiate_from_opt(opt=self.quantizer_opt)
        self.criterion= VQLoss(beta=vq_beta)
    def encode(self, Xbd, **kwargs):
        grid_feat, grid_mask = self.encoder(Xbd/2.) # [-1,1] -> [-.5,.5]
        return grid_feat, grid_mask
    def encode_quant(self, Xbd, **kwargs):
        grid_feat, grid_mask = self.encode(Xbd, **kwargs)
        if self.quantizer_opt is not None:
            quant_feat, quant_feat_st, quant_ind, quant_diff  = self.quantizer(grid_feat)
            grid_feat = quant_feat_st # do not update the weight of the codebook of quantizer
        else:
            quant_feat, quant_feat_st, quant_ind, quant_diff = None, None, None, None
        
        return dict(quant_feat=quant_feat_st, quant_ind=quant_ind, quant_diff=quant_diff, grid_mask=grid_mask)
    def quantize_cloud(self, cloud):
        encoded = self.encode_quant(cloud)
        mask = encoded["grid_mask"]
        mode = pth_get_mode(encoded["quant_ind"].reshape(-1))
        #print("MODE: ", mode, mask.sum(), (encoded["quant_ind"]!=mode).sum())
        quant_ind = torch.zeros_like(encoded["quant_ind"]).type_as(encoded["quant_ind"]) + mode
        quant_ind[mask] = encoded["quant_ind"][mask]
        return quant_ind, mode, encoded
    def decode(self, grid_feat, Xtg=None, **kwargs):
        # quant_feat: (res, res, res)
        max_limit = 256**3
        if Xtg.shape[1] > max_limit:
            with torch.no_grad():
                batch_xtg = torch.split(Xtg, max_limit, dim=1)
                b_ytgs = []
                for b_xtg in batch_xtg:
                    b_ytgs.append( self.decoder(b_xtg/2, grid_feat) )
                Ytg_logits = torch.cat(b_ytgs, dim=1)
        else:
            Ytg_logits = self.decoder(Xtg/2., grid_feat) # [-1,1] -> [-.5,.5]
        return dict(logits=Ytg_logits)
    def decode_index(self, code_ind, Xtg):
        quant_feat = self.quantizer.get_code(code_ind)
        return self.decode(quant_feat, Xtg)
    def forward(self, Xbd, Xtg, **kwargs):
        # Xbd: [B, num_pts, x_dim], Xtg: [B, num_probes, x_dim]
        B, num_pts, x_dim = Xbd.shape
        # (B, C, res, res, res)
        grid_feat, grid_mask = self.encode(Xbd)
        if self.quantizer_opt is not None:
            quant_feat, quant_feat_st, quant_ind, quant_diff  = self.quantizer(grid_feat)
            grid_feat = quant_feat_st # do not update the weight of the codebook of quantizer
        else:
            quant_feat, quant_feat_st, quant_ind, quant_diff = None, None, None, None
        logits = self.decode(grid_feat, Xtg=Xtg)["logits"]
        
        return dict(logits=logits, quant_feat=quant_feat_st, quant_ind=quant_ind, quant_diff=quant_diff, grid_mask=grid_mask)

    def get_loss(self, batch, batch_idx, stage="train"):
        Xbd = batch["Xbd"] if self.Xct_as_Xbd==False else batch["Xct"]
        out = self.forward(Xbd=Xbd, Xtg=batch["Xtg"])
        losses = self.criterion.get_loss(logits=out["logits"], label=batch["Ytg"], quant_diff=out["quant_diff"])
        return losses
    def training_step(self, batch, batch_idx, stage="train"):
        losses = self.get_loss(batch, batch_idx, stage="train")
        for loss_name in losses:
            self.log(f'{stage}/{loss_name}', losses[loss_name], prog_bar=True, sync_dist=True)
        return losses['loss']
    def validation_step(self, batch, batch_idx=0, stage='val', return_data=False):
        losses = self.get_loss(batch, batch_idx, stage="val")
        for loss_name in losses:
            self.log(f'{stage}/{loss_name}', losses[loss_name], prog_bar=False, sync_dist=True)
        self.log('val_loss', losses['loss'], prog_bar=False, sync_dist=True)
        return losses['loss']
    def test_step(self, batch, batch_idx=0, stage='test', return_data=False):
        losses = self.get_loss(batch, batch_idx, stage="test")
        for loss_name in losses:
            self.log(f'{stage}/{loss_name}', losses[loss_name], prog_bar=False, sync_dist=True)
        return losses['loss']
    def configure_optimizers(self):
        oopt = self.optim_opt
        if oopt is None:
            return [], []
        optim = torch.optim.Adam(self.parameters(), lr=self.optim_opt['lr'])

        sched_name = oopt['scheduler']
        if sched_name   == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optim, \
                                        step_size=oopt['step_size'], \
                                        gamma=oopt['gamma'])
        elif sched_name == 'None':
            return optim
        else:
            raise NotImplementedError(f'Can not use scheduler:{sched_name}')
        return [optim], [scheduler]
    

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
            losses = dict(loss=self.loss, recon_loss=self.recon_loss, diff_loss=self.diff_loss)
        else:
            self.loss = self.recon_loss
            losses = dict(loss=self.loss, recon_loss=self.recon_loss)
        return losses

class VisSparseRecon3D(plutil.VisCallback):
    def __init__(self, samples=32, Xct_as_Xbd=False, quant_grid_depth=4, decoder_resolution=128, vocab_size=4096, max_length=512, end_tokens=(4096,4096), resolution=(512,512), vis_Ytg=True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__dict__.update(locals())
        #self.visual_indices = list(range(100000))
        self.vis_camera = dict(camPos=np.array([2,2,2]), camLookat=np.array([0.,0.,0.]),\
                camUp=np.array([0,1,0]),camHeight=2,resolution=resolution, samples=samples)
        self.cloudR = 0.008
        self.all_Xtg = torch.from_numpy(nputil.makeGrid([-1,-1,-1.],[1.,1,1],[decoder_resolution,]*3, indexing="ij"))[None,...]
    def decode_batch(self, quant_ind, batch):
        # quant_ind (1, res, res, res)
        sparse, mode  = batch_dense2sparse(quant_ind, max_length=self.max_length, end_tokens=self.end_tokens)
        packed_sparse = pack_sparse( sparse, end_tokens=self.end_tokens)
        dense = batch_sparse2dense(packed_sparse, empty_ind=mode, dense_res=2**self.quant_grid_depth)

        decoded = self.pl_module.decode_index(dense, self.all_Xtg.type_as(batch["Xtg"]))
        if type(decoded) is not dict:
            decoded = dict(logits=decoded)
        decoded = ptutil.ths2nps(decoded)
        ret = {"logits":decoded["logits"], "quant_ind":encoded["quant_ind"], "sparse":packed_sparse, "batch":batch}
        return ptutil.ths2nps(ret)

    def compute_batch(self, batch, input_name=""):
        # out = self.pl_module(batch["Xbd"], self.all_Xtg.type_as(batch["Xtg"]))
        # if type(out) is not dict:
        #     out = dict(logits=out)
        # return {"logits":out["logits"], "batch":batch}
        
        Xbd = batch["Xbd"] if ("Xbd" in batch and self.Xct_as_Xbd==False) else batch["Xct"]
        quant_ind, mode, encoded = self.pl_module.quantize_cloud(Xbd)
        grid_mask = encoded["grid_mask"]
        #quant_ind[quant_ind> =1288]=1391

        #print( torch.stack(torch.unique(quant_ind.reshape(-1), return_counts=True),axis=-1) )
        sparse, mode  = batch_dense2sparse(quant_ind, max_length=self.max_length, end_tokens=self.end_tokens)
        packed_sparse = pack_sparse( sparse, end_tokens=self.end_tokens)
        dense = batch_sparse2dense( packed_sparse, empty_ind=mode, dense_res=2**self.quant_grid_depth)

        decoded = self.pl_module.decode_index(dense, self.all_Xtg.type_as(batch["Xbd"]))
        if type(decoded) is not dict:
            decoded = dict(logits=decoded)
        decoded = ptutil.ths2nps(decoded)
        ret = { "logits":decoded["logits"], "quant_ind":encoded.get("quant_ind"), 
                "sparse":packed_sparse,     "grid_mask":grid_mask, "batch":batch}
        return ptutil.ths2nps(ret)

    def visualize_batch(self, computed, input_name=""):
        computed = ptutil.ths2nps(computed)
        batch, logits, quant_ind, sparse_quant_ind = computed["batch"], computed["logits"], computed["quant_ind"], computed["sparse"]
        quant_ind = convonet_to_nnrecon(quant_ind)
        occupancy = nputil.sigmoid(logits.reshape(-1))
        Xtg = batch['Xtg'][0] if "Xtg" in batch else None
        all_Xtg = self.all_Xtg.numpy()
        imgs = {}
        if 'Ytg' in batch and self.vis_Ytg==True:
            imgs["gt"] = npfvis.plot_3d_recon(Xtg=Xtg, Ytg=batch['Ytg'][0], camera_kwargs=self.vis_camera)
        if "Xbd" in batch:
            imgs["gt_pc"] = fresnelvis.renderMeshCloud(cloud=batch["Xbd"][0], cloudR=self.cloudR, **self.vis_camera)
        if "Xct" in batch:
            partial_cloud  = fresnelvis.renderMeshCloud(cloud=batch["Xct"][0], cloudR=self.cloudR, **self.vis_camera)
            imgs.update( {"data_pc_p":partial_cloud} )
        #imgs["recon"] = npfvis.plot_3d_sample(Xct=batch["Xbd"][0], Yct=None, Xtg=batch['Xtg'][0], Ytg=None, pred_y=occupancy[0], show_images=["pred"])[0]
        #imgs["recon"] = npfvis.plot_3d_sample(Yct=None, Xtg=batch['Xtg'][0], Ytg=None, pred_y=occupancy[0], show_images=["pred"])[0]
        imgs["recon"] = npfvis.plot_3d_recon(Xtg=all_Xtg, Ytg=occupancy, camera_kwargs=self.vis_camera)
        
        
        pos_ind, val_ind = sparse_quant_ind[:,1], sparse_quant_ind[:,2]
        pos_ind = sparse_convonet_to_nnrecon(pos_ind, shape=(2**self.quant_grid_depth,)*3)
        imgs["quant_ind"] = vis3d.IndexVoxelPlot( pos_ind, val_ind, val_max=self.vocab_size, depth= self.quant_grid_depth, camera_kwargs=self.vis_camera)
        pos_ind = sparse_convonet_to_nnrecon( computed["grid_mask"][0].reshape(-1).nonzero()[0], shape=(2**self.quant_grid_depth,)*3 )
        imgs["mask_ind"] = vis3d.IndexVoxelPlot( pos_ind, pos_ind, val_max=4096, depth= self.quant_grid_depth, camera_kwargs=self.vis_camera)
        return imgs

def codebook_nonzero_count(pl_model):
    codebook = pl_model.quantizer
    #codebook = pl_model.model.codebook
    print(len(codebook.N.nonzero()))
    print(len((codebook.z_avg.abs().sum(axis=-1)>7.1186e-10).nonzero()))
    w = codebook.embedding.weight
    print(w.shape)
    print(len((w.abs().sum(axis=-1)>10e-1).nonzero()))

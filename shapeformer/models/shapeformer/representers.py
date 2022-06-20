""" Standard output format: raveled coords
"""
import os
import math
from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import time
from xgutils import *
from einops import rearrange, repeat
import numpy as np
import yaml

from .common import *


class Representer(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def get_indices(self, **kwargs):
        pass

    def get_extra_indices(self, c_indices, z_indices):
        pass

    def get_output_indices(self, indices):
        pass

    def sampling_masker(self, logits, idx, extra_idx=None, L_cond=None, step_j=None, tuple_i=None):
        return logits

    def init_trained_model_from_ckpt(self, config):
        # ckpt = torch.load(config["ckpt_path"])
        # if hasattr(config, "yaml_path"):  # update path in ckpt with a yaml config
        #     with open(config["yaml_path"]) as file:
        #         yamlf = yaml.load(file, Loader=yaml.FullLoader)
        #     sysutil.dictUpdate(ckpt["hyper_parameters"],
        #                        yamlf["pl_model_opt"]["kwargs"])

        model = sysutil.load_object(
            config["class"]).load_from_checkpoint(config["ckpt_path"])
        model = model.eval()
        # freeze model, so that its not in grad flow, neccessary for ddp
        model.requires_grad_(False)
        model.train = disabled_train
        return model

# raveled coord, no extra inds


class ShapeRepresenter(Representer):
    def __init__(self,  voxel_res=16, end_tokens=None, input_end_tokens=None, block_size=None,
                 uncond=False, no_val_ind=False, vqvae_opt=None, cloud_shrinkage=1.,
                 random_cind_masking=False,  # random_cind_permutation=
                 mask_invalid=True, mask_invalid_completion=False):
        super().__init__()
        self.__dict__.update(locals())
        self.end_tokens = end_tokens
        if input_end_tokens is None:
            self.input_end_tokens = self.end_tokens
        else:
            self.input_end_tokens = input_end_tokens
        self.max_length = self.block_size // 2
        self.vqvae_model = self.init_trained_model_from_ckpt(vqvae_opt)

    @torch.no_grad()
    def encode_cloud(self, cloud):
        with torch.no_grad():
            quant_ind, mode, encoded = self.vqvae_model.quantize_cloud(
                cloud * self.cloud_shrinkage if hasattr(self, "cloud_shrinkage") else 1.)
            sparse_unpacked, mode = batch_dense2sparse(
                quant_ind, max_length=self.max_length, end_tokens=self.input_end_tokens)
            if self.no_val_ind == True:
                sparse_unpacked[:, :, -1] *= 0
        return encoded["quant_feat"], quant_ind, mode, sparse_unpacked

    def get_indices(self, Xct, Xbd=None, stage="train", **kwargs):
        _, _, mode1, c_indices = self.encode_cloud(Xct)
        if Xbd is None:
            z_indices = c_indices[:, :0, :]
        else:
            _, _, _, z_indices = self.encode_cloud(Xbd)
        if self.uncond == True:
            B, _, tuple_n = c_indices.shape
            c_indices = torch.zeros(
                B, 1, tuple_n)+torch.tensor(self.input_end_tokens)[None, None, :]
            c_indices = c_indices.type_as(Xct).long()
        others = dict(empty_index=mode1, origin_c_indices=c_indices,
                      origin_z_indices=z_indices)

        if stage == "train" and self.random_cind_masking == True and c_indices.shape[1] >= 1:
            max_num = c_indices.shape[1] - 1
            select_num = np.random.randint(0, max_num+1)
            selected_ind = np.sort(np.random.choice(
                max_num, select_num, replace=False))
            c_indices = torch.cat(
                [c_indices[:, selected_ind, :], c_indices[:, -1:, :]], axis=1)

        extra_indices = self.get_extra_indices(c_indices, z_indices)
        c_indices, z_indices = self.convert_input_indices(c_indices, z_indices)
        return c_indices, z_indices, extra_indices, others

    def get_extra_indices(self, c_indices, z_indices):
        cz_indices = torch.cat([c_indices, z_indices], axis=1)
        # no extra indices
        B, L = cz_indices.shape[:2]
        extra_indices = torch.zeros(B, L, 1).type_as(cz_indices)
        return extra_indices

    def convert_input_indices(self, c_indices, z_indices):
        # transform input indices
        return c_indices, z_indices

    def convert_output_indices(self, indices):
        # transform output indices
        return indices

    def sampling_masker(self, logits, idx, extra_idx=None, L_cond=None, step_j=None, tuple_i=None):
        """ logits: (B, vocab_size), idx[-1]: current position (to be sampled)"""
        # only apply to position index
        logits = logits.clone()
        latest_positions = idx[:, -2, 0]
        B = logits.shape[0]
        end_tokens = torch.tensor(self.end_tokens).type_as(idx)
        if tuple_i == 1:
            # if pos==end_token[0], then val should be end_token[1]
            end_mask = (idx[:, -1, 0] == end_tokens[0])
            logits[end_mask, :] = -float('Inf')
            logits[end_mask, end_tokens[1]] = 1.
            return logits
        positions = torch.arange(logits.shape[-1])[None, :].type_as(idx)
        if self.mask_invalid and step_j > 0:
            # print("masking")
            # the latest sampled position is the largest
            # only keep the possibilities for index>largest, except for end token
            invalid_mask = positions <= latest_positions[:, None]
            invalid_mask[:, end_tokens[0]] = False
            logits[invalid_mask] = -float('Inf')  # mask out invalids
        if self.mask_invalid_completion:
            cond_pos_idx = idx[:, :L_cond, 0].contiguous()
            # (B, L_cond+1) append 1+end_tokens[0] to prevent corner cases
            cond_poses = torch.cat(
                (cond_pos_idx, 1 + end_tokens[0][None, None].expand(B, -1)), axis=1)
            # find next position of condition
            next_ids = torch.searchsorted(
                cond_poses, latest_positions[:, None], right=True)
            next_poses = torch.gather(cond_poses, dim=1, index=next_ids)
            #print("Next poses", next_ids)
            #print("Current seq", idx[:,L_cond:L_cond+step_j,0])
            # mask the logits after the next cond
            invalid_mask = positions > next_poses
            logits[invalid_mask] = -float('Inf')
        return logits

# category conditioning


class CC(ShapeRepresenter):
    def get_indices(self, Xct, Xbd, **kwargs):
        _, _, mode1, c_indices = self.encode_cloud(Xct)
        _, _, mode2, z_indices = self.encode_cloud(Xbd)
        if self.uncond == True:
            B, L, tuple_n = z_indices.shape
            c_indices = torch.zeros(
                B, 1, tuple_n)+torch.tensor(self.input_end_tokens)[None, None, :]
            c_indices = c_indices.type_as(z_indices)
        others = dict(empty_index=mode1, origin_c_indices=c_indices,
                      origin_z_indices=z_indices)
        extra_indices = self.get_extra_indices(c_indices, z_indices)
        c_indices, z_indices = self.convert_input_indices(c_indices, z_indices)
        return c_indices, z_indices, extra_indices, others

# AR: absolute raveled
# RR: relative raveled
# AU: absolute unraveled
# ...
# N: next condition pos
# X_Y: generate X while adding Y as additional information


class AR(ShapeRepresenter):
    pass


class AR_N(ShapeRepresenter):
    def get_extra_indices(self, c_indices, z_indices):
        L_c, L_z = c_indices.shape[1], z_indices.shape[1]

        c_extra = c_indices[..., 0].clone()
        z_extra = get_next_cond(
            c_indices[..., 0], z_indices[..., 0], self.end_tokens[0])
        extra_indices = torch.cat([c_extra, z_extra], axis=1)[..., None]
        # print("!",extra_indices)
        return extra_indices


class AR_RR(ShapeRepresenter):  # checked
    def get_extra_indices(self, c_indices, z_indices):
        L_c, L_z = c_indices.shape[1], z_indices.shape[1]
        cRR = AR_to_RR(c_indices[..., 0], end_token=self.end_tokens[0])
        zRR = AR_to_RR(z_indices[..., 0], end_token=self.end_tokens[0])
        extra_indices = torch.cat([cRR, zRR], axis=1)[..., None]
        # print("!",extra_indices)
        return extra_indices


class RR(ShapeRepresenter):  # checking TODO
    def convert_input_indices(self, c_indices, z_indices):
        # transform input indices
        c_indices, z_indices = c_indices.clone(), z_indices.clone()
        c_indices[..., 0] = AR_to_RR(c_indices[..., 0], self.end_tokens[0])
        z_indices[..., 0] = AR_to_RR(z_indices[..., 0], self.end_tokens[0])
        return c_indices, z_indices

    def convert_output_indices(self, indices):
        # transform output indices
        indices = indices.clone()
        indices[..., 0] = RR_to_AR(indices[..., 0], self.end_tokens[0])
        return indices

    def sampling_masker(self, logits, idx, extra_idx=None, L_cond=None, step_j=None, tuple_i=None):
        """ logits: (B, vocab_size), idx[-1]: current position (to be sampled)"""
        # return logits
        logits = logits.clone()
        latest_positions = idx[:, -2, 0]
        end_tokens = torch.tensor(self.end_tokens)
        if tuple_i == 0:
            if step_j > 0:
                dead_mask = idx[:, -2, 0] == end_tokens[0]
            else:
                dead_mask = idx[:, -1, 0] != idx[:, -1, 0]  # all alive
        if tuple_i > 0:
            dead_mask = idx[:, -1, 0] == end_tokens[0]
        alive_mask = torch.logical_not(dead_mask)
        # masking dead sequences
        logits[dead_mask, :] = -float('Inf')
        logits[dead_mask,  end_tokens[tuple_i]] = 1.
        # masking alive sequences
        alive_logits = logits[alive_mask]
        alive_idx = idx[alive_mask]
        positions = torch.arange(
            alive_logits.shape[-1])[None, :].type_as(alive_idx)
        if tuple_i > 0:  # sequence only dies at position index
            logits[alive_mask, end_tokens[tuple_i]] = -float('Inf')
        if tuple_i == 0 and alive_idx.shape[0] > 0:
            if step_j > 0:
                prev_pos = alive_idx[:, L_cond:-1, 0].sum(axis=-1)
                #print("alive", alive_idx.shape[0], "prev_pos", prev_pos[0])
                max_pos = self.voxel_res**3 - 1
                feasible = max_pos - prev_pos
                #print("!!", feasible[0])
                # should not exceed the max
                invalid_mask = positions > feasible[:, None]
                invalid_mask[:, 0] = True  # the sequence should be monotonic
                # Always has the probability to end the sequence
                invalid_mask[:, end_tokens[0]] = False
            else:
                invalid_mask = torch.zeros_like(alive_logits).bool()
                # should not produce 0-length sequence
                invalid_mask[:, end_tokens[0]] = True
            assert invalid_mask.all(
                axis=-1).any() == False, "There is a sequence has no valid logits!"
            alive_logits[invalid_mask] = -float('Inf')
        logits[alive_mask] = alive_logits
        return logits


class RR_AR(RR):
    def get_extra_indices(self, c_indices, z_indices):
        # AR is the default repr.
        extra_indices = torch.cat(
            [c_indices[..., 0], z_indices[..., 0]], axis=1)[..., None]
        return extra_indices


class AU(ShapeRepresenter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.AR_end_token = self.voxel_res**3
        self.AU_end_token = self.voxel_res

    def convert_input_indices(self, c_indices, z_indices):
        # transform input indices
        nci = AR_to_AU(c_indices[..., 0:1], self.AR_end_token,
                       self.AU_end_token, reso=self.voxel_res)
        nzi = AR_to_AU(z_indices[..., 0:1], self.AR_end_token,
                       self.AU_end_token, reso=self.voxel_res)
        nci = torch.cat([nci, c_indices[..., 1:2]], axis=-1)
        nzi = torch.cat([nzi, z_indices[..., 1:2]], axis=-1)
        return nci, nzi

    def convert_output_indices(self, indices):
        # transform output indices
        ni = AU_to_AR(indices[..., :-1], self.AR_end_token,
                      self.AU_end_token, reso=self.voxel_res)
        ni = torch.cat([ni, indices[..., -1:]], axis=-1)
        return ni

    def sampling_masker(self, logits, idx, extra_idx=None, L_cond=None, step_j=None, tuple_i=None):
        """ logits: (B, vocab_size), idx[-1]: current position (to be sampled)"""
        logits = logits.clone()
        latest_positions = idx[:, -2, 0]
        end_tokens = torch.tensor(self.end_tokens)
        # recognize which sequence is dead
        if tuple_i == 0:
            if step_j > 0:
                dead_mask = idx[:, -2, 0] == end_tokens[0]
            else:
                dead_mask = idx[:, -1, 0] != idx[:, -1, 0]  # all alive
        if tuple_i > 0:
            dead_mask = idx[:, -1, 0] == end_tokens[0]
        alive_mask = torch.logical_not(dead_mask)
        # masking dead sequences
        logits[dead_mask, :] = -float('Inf')
        logits[dead_mask,  end_tokens[tuple_i]] = 1.
        # masking alive sequences
        if alive_mask.sum() == 0:
            return logits
        alive_logits = logits[alive_mask]
        alive_idx = idx[alive_mask]
        positions = torch.arange(
            alive_logits.shape[-1])[None, :].type_as(alive_idx)
        if step_j > 0:
            prev_z, prev_y, prev_x = alive_idx[:, -2,
                                               0], alive_idx[:, -2, 1], alive_idx[:, -2, 2]
            if tuple_i == 0:  # z coord
                #print("!!", feasible[0])
                # z coord should be non-decreasing
                invalid_mask = positions < prev_z[:, None]
                # if previously y & x are at maximum value, then z must increase at this step
                corner_case = torch.logical_and(
                    prev_y == self.voxel_res-1, prev_x == self.voxel_res-1)
                invalid_mask[corner_case, prev_z[corner_case]] = True

                # Always has the probability to end the sequence
                invalid_mask[:, end_tokens[0]] = False
            if tuple_i == 1:  # y coord
                # y should be non-decreasing if current z == previous z
                cur_z = alive_idx[:, -1, 0]
                invalid_mask = positions < prev_y[:, None]

                # if previously y & x are at maximum value, then z must increase at this step
                corner_case = prev_x == self.voxel_res-1
                invalid_mask[corner_case, prev_y[corner_case]] = True

                invalid_mask[cur_z != prev_z] = False

            if tuple_i == 2:  # x coord
                # x should be increasing if current z == previous z and current y == previous y
                cur_z, cur_y = alive_idx[:, -1, 0], alive_idx[:, -1, 1]
                invalid_mask = positions <= prev_x[:, None]
                inapplicable_mask = torch.logical_or(
                    cur_z != prev_z, cur_y != prev_y)
                invalid_mask[inapplicable_mask] = False
            if tuple_i == 3:  # val index
                invalid_mask = torch.zeros_like(alive_logits).bool()
        else:
            invalid_mask = torch.zeros_like(alive_logits).bool()
            # should not produce 0-length sequence
            invalid_mask[:, end_tokens[tuple_i]] = True
        if tuple_i > 0:  # sequence only dies at position index z
            invalid_mask[:, end_tokens[tuple_i]] = True
        #print("!",step_j, tuple_i,idx)
        assert invalid_mask.all(
            axis=-1).any() == False, f"There is a sequence has no valid logits! step&tuple: {step_j}, {tuple_i}"
        alive_logits[invalid_mask] = -float('Inf')
        logits[alive_mask] = alive_logits
        return logits


def ravel_index(indices, reso):
    # (*, 3) -> (*, 1)
    # (z, y, x) -> i
    ri = indices[..., 2:3] + reso * \
        (indices[..., 1:2] + reso * indices[..., 0:1])
    return ri


def unravel_index(indices, reso):
    # (*, 1) -> (*, 3)
    # i -> (z, y, x)
    xi = indices % reso
    yi = (indices // reso) % reso
    zi = indices // reso // reso
    ui = torch.cat([zi, yi, xi], axis=-1)
    return ui


def AR_to_RR(AR, end_token):
    """ AR:(B, L) """
    if AR.shape[1] == 0:
        return AR
    diff = AR - torch.roll(AR, 1, 1)
    diff[AR == end_token] = end_token
    diff[:, 0] = AR[:, 0]
    return diff


def RR_to_AR(RR, end_token):
    """ AR:(B, L) """
    if RR.shape[1] == 0:
        return RR
    AR = RR.cumsum(axis=1)
    AR[RR == end_token] = end_token
    return AR


def AR_to_AU(AR, AR_end_token, AU_end_token, reso=16):
    """ AR:(B, L, 1) """
    AR = AR.clone()
    AU = torch.zeros(AR.shape[0], AR.shape[1], 3).type_as(AR)
    end_mask = (AR == AR_end_token).squeeze(-1)
    val_mask = (AR != AR_end_token).squeeze(-1)
    AU[end_mask] = AU_end_token  # 16 is the end token for z,y,x
    AU[val_mask] = unravel_index(AR[val_mask], reso=reso)
    return AU


def AU_to_AR(AU, AR_end_token, AU_end_token, reso=16):
    """ AU:(B, L, 3) """
    AU = AU.clone()
    AR = torch.zeros(AU.shape[0], AU.shape[1], 1).type_as(AU)
    end_mask = (AU == AU_end_token).any(axis=-1)  # (B, L)
    val_mask = torch.logical_not(end_mask)
    AR[end_mask] = AR_end_token  # 16 is the end token for x,y,z
    AR[val_mask] = ravel_index(AU[val_mask], reso=reso)
    return AR


def get_next_cond(c_pos_indices, z_pos_indices, end_token):
    # find next position of condition
    next_ids = torch.searchsorted(c_pos_indices, z_pos_indices, right=True)
    next_ids[z_pos_indices == end_token] = c_pos_indices.shape[1]-1
    next_cond_pos = torch.gather(c_pos_indices, dim=1, index=next_ids)
    if (z_pos_indices.shape[1] == 0):
        return z_pos_indices.clone()

    next_cond_pos[z_pos_indices == end_token] = end_token
    # print(next_cond_pos)
    return next_cond_pos

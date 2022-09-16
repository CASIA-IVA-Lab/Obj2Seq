# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Anchor DETR (https://github.com/megvii-research/AnchorDETR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import numpy as np

import torch
from torch import nn

from util.misc import NestedTensor

from .encoder import build_encoder
from .prompt_indicator import PromptIndicator
from .object_decoder import ObjectDecoder
from .position_encoding import build_position_encoding

class Transformer(nn.Module):
    def __init__(self, args=None): # MODEL
        super().__init__()
        self.d_model = args.hidden_dim
        self.encoder = build_encoder(args) if args.enc_layers > 0 else None
        self.position_embed = build_position_encoding(args)
        if self.encoder is not None:
            self.level_embed = nn.Parameter(torch.Tensor(args.num_feature_levels, self.d_model))
            nn.init.normal_(self.level_embed)

        # prompt_indicator
        if args.with_prompt_indicator:
            self.prompt_indicator = PromptIndicator(args.PROMPT_INDICATOR)
        else:
            self.prompt_indicator = None

        # object decoder
        if args.with_object_decoder:
            self.object_decoder = ObjectDecoder(self.d_model, args=args.OBJECT_DECODER)
        else:
            self.object_decoder = None

    def forward(self, srcs, masks, targets=None):
        # srcs: a list of tensors [(bs, c, h_i, w_i)]
        # masks: a list of tensors [(bs, h_i, w_i)]
        # targets:
        bs = srcs[0].shape[0]
        srcs, mask, enc_kwargs, cls_kwargs, obj_kwargs = self.prepare_for_deformable(srcs, masks)

        srcs = self.encoder(srcs, padding_mask=mask, **enc_kwargs) if self.encoder is not None else srcs
        outputs, loss_dict = {}, {}

        if self.prompt_indicator is not None:
            cls_outputs, cls_loss_dict = self.prompt_indicator(srcs, mask, targets=targets, kwargs=cls_kwargs)
            outputs.update(cls_outputs)
            loss_dict.update(cls_loss_dict)
            additional_object_inputs = dict(
                bs_idx = outputs["bs_idx"] if "bs_idx" in outputs else None,
                cls_idx = outputs["cls_idx"] if "cls_idx" in outputs else None,
                class_vector = outputs["tgt_class"], # cs_all, d
                previous_logits = outputs["cls_label_logits"], # bs, 80
            )
        else:
            additional_object_inputs = {}

        if self.object_decoder is not None:
            obj_outputs, obj_loss_dict = self.object_decoder(srcs, mask, targets=targets, additional_info=additional_object_inputs, kwargs=obj_kwargs)
            outputs.update(obj_outputs)
            loss_dict.update(obj_loss_dict)

        return outputs, loss_dict

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def prepare_for_deformable(self, srcs, masks):
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask) in enumerate(zip(srcs, masks)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            if self.encoder is not None:
                pos_embed = self.position_embed(NestedTensor(src, mask)).to(src.dtype)
                pos_embed = pos_embed.flatten(2).transpose(1, 2)
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) if self.encoder is not None else None
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        reference_points_enc = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        enc_kwargs = dict(spatial_shapes = spatial_shapes,
                          level_start_index = level_start_index,
                          reference_points = reference_points_enc,
                          pos = lvl_pos_embed_flatten)
        cls_kwargs = dict(src_level_start_index=level_start_index)
        obj_kwargs = dict(src_spatial_shapes=spatial_shapes,
                          src_level_start_index=level_start_index,
                          src_valid_ratios=valid_ratios)
        return src_flatten, mask_flatten, enc_kwargs, cls_kwargs, obj_kwargs

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
import copy

import torch
import torch.nn.functional as F
from torch import nn

from models.ops.modules import MSDeformAttn


class BasicEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.hidden_dim
        self.normalize_before = args.pre_norm
        self.build_self_attn(args)

        # self attention
        self.dropout1 = nn.Dropout(args.dropout)
        self.norm1 = nn.LayerNorm(self.d_model)

        # ffn
        self.ffn = FFN(self.d_model, args.dim_feedforward, args.dropout, args.activation, normalize_before=self.normalize_before)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, **kwargs):
        # self attention
        src2 = self.self_attn_forward(src, **kwargs)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.ffn(src)
        return src

    def forward_pre(self, src, **kwargs):
        # self attention
        src2 = self.norm1(src)
        src2 = self.self_attn_forward(src2, **kwargs)

        src = src + self.dropout1(src2)

        # ffn
        src = self.ffn(src)
        return src

    def forward(self, *args, **kwargs):
        if self.normalize_before:
            return self.forward_pre(*args, **kwargs)
        return self.forward_post(*args, **kwargs)


class DeformableEncoderLayer(BasicEncoderLayer):
    def build_self_attn(self, args):
        self.self_attn = MSDeformAttn(self.d_model, args.n_levels, args.nheads, args.n_points)

    def self_attn_forward(self, src, **kwargs):
        pos = kwargs.pop("pos", None)
        reference_points = kwargs.pop("reference_points")
        spatial_shapes = kwargs.pop("spatial_shapes")
        level_start_index = kwargs.pop("level_start_index")
        padding_mask = kwargs.pop("padding_mask", None)
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        return src2


class BasicDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model = args.hidden_dim
        self.n_heads = args.nheads
        self.normalize_before = args.pre_norm

        # cross attention
        self.build_cross_attn(args)
        self.dropout1 = nn.Dropout(args.dropout)
        self.norm1 = nn.LayerNorm(self.d_model)

        # self attention
        self.self_attn = not args.no_self_attn
        if self.self_attn:
            self.self_attn = nn.MultiheadAttention(self.d_model, self.n_heads, dropout=args.dropout)
            self.dropout2 = nn.Dropout(args.self_attn_dropout)
            self.norm2 = nn.LayerNorm(self.d_model)

        # ffn
        self.ffn = FFN(self.d_model, args.dim_feedforward, args.dropout, args.activation, normalize_before=self.normalize_before)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def self_attn_forward(self, tgt, query_pos, **kwargs):
        if query_pos is not None and query_pos.shape[0] != tgt.shape[0]:
            cs = tgt.shape[0] // query_pos.shape[0]
            query_pos_self = query_pos.repeat_interleave(repeats=cs, dim=0)
        else:
            query_pos_self = query_pos
        q = k = self.with_pos_embed(tgt, query_pos_self)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        return tgt2

    def forward_post(self, tgt, query_pos, **kwargs):
        # self attention
        if self.self_attn:
            tgt2 = self.self_attn_forward(tgt, query_pos, **kwargs)
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        tgt2 = self.cross_attn_forward(tgt, query_pos, **kwargs)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.ffn(tgt)

        return tgt

    def forward_pre(self, tgt, query_pos, **kwargs):
        # self attention
        if self.self_attn:
            tgt2 = self.norm2(tgt)
            tgt2 = self.self_attn_forward(tgt2, query_pos, **kwargs)
            tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm1(tgt)
        tgt2 = self.cross_attn_forward(tgt2, query_pos, **kwargs)
        tgt = tgt + self.dropout1(tgt2)

        # ffn
        tgt = self.ffn(tgt)

        return tgt

    def forward(self, *args, **kwargs):
        if self.normalize_before:
            return self.forward_pre(*args, **kwargs)
        return self.forward_post(*args, **kwargs)


class MultiHeadDecoderLayer(BasicDecoderLayer):
    def build_cross_attn(self, args):
        self.cross_attn = nn.MultiheadAttention(self.d_model, self.n_heads, dropout=args.dropout)

    def cross_attn_forward(self, tgt, query_pos, **kwargs):
        # tgt: 
        # query_pos: 
        # srcs: bs, h, w, c
        bs_all, seq, c = tgt.shape
        srcs = kwargs["srcs"]
        bs = srcs.shape[0]
        if bs_all > bs:
            tgt = tgt.view(bs, -1, c)
            cs = bs_all // bs

        src_padding_masks = kwargs.pop("src_padding_masks")
        posemb_2d = kwargs.pop("posemb_2d", 0)
        query_pos = torch.zeros_like(tgt) if query_pos is None else query_pos.repeat(1,cs,1)
        tgt2 = self.cross_attn((tgt + query_pos).transpose(0, 1),
                                (srcs + posemb_2d).reshape(bs, -1, c).transpose(0,1),
                                srcs.reshape(bs, -1, c).transpose(0, 1), key_padding_mask=src_padding_masks.reshape(bs, -1))[0].transpose(0,1)
        return tgt2.reshape(bs_all, seq, c)

    def forward(self, tgt, query_pos, reference_points, srcs, src_padding_masks, **kwargs):
        return super().forward(tgt, query_pos, srcs=srcs, src_padding_masks=src_padding_masks, **kwargs)


class DeformableDecoderLayer(BasicDecoderLayer):
    def build_cross_attn(self, args):
        self.cross_attn = MSDeformAttn(self.d_model, args.n_levels, args.nheads, args.n_points, no_value_proj=args.cross_attn_no_value_proj)

    def cross_attn_forward(self, tgt, query_pos, reference_points, srcs, src_padding_masks, **kwargs):
        # tgt: bs_all, seq, c
        # src: bs, seq_src, c
        # reference_points: bs / bs_all, seq, lvl, 2 or 4 (len_pt)
        bs_all, seq, c = tgt.shape
        num_levels = reference_points.shape[-2]
        bs = srcs.shape[0]
        cs_batch = kwargs.pop("cs_batch", None)
        src_spatial_shapes = kwargs.pop("src_spatial_shapes")
        level_start_index = kwargs.pop("src_level_start_index")

        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               srcs, src_spatial_shapes, level_start_index, src_padding_masks, cs_batch=cs_batch)
        return tgt2

    def forward(self, tgt, query_pos, reference_points, srcs, src_padding_masks, **kwargs):
        # reference_points: bs / bs_all, seq, 2 or 4
        src_valid_ratios = kwargs.pop("src_valid_ratios") # bs, level, 2
        if reference_points.shape[-1] == 4:
            src_valid_ratios = torch.cat([src_valid_ratios, src_valid_ratios], dim=-1)
        # if the number of reference_points and number of src_valid_ratios not match.
        # Expand and repeat for them
        if src_valid_ratios.shape[0] != reference_points.shape[0]:
            repeat_times = (reference_points.shape[0] // src_valid_ratios.shape[0])
            src_valid_ratios = src_valid_ratios.repeat_interleave(repeat_times, dim=0)
        src_valid_ratios = src_valid_ratios[:, None] if reference_points.dim() == 3 else src_valid_ratios[:, None, None]
        reference_points_input = reference_points[..., None, :] * src_valid_ratios
        return super().forward(tgt, query_pos, reference_points=reference_points_input, srcs=srcs, src_padding_masks=src_padding_masks, **kwargs)


class FFN(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024, dropout=0., activation='relu', normalize_before=False):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.normalize_before = normalize_before

    def forward_post(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src):
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src2))))
        src = src + self.dropout3(src2)
        return src

    def forward(self, src):
        if self.normalize_before:
            return self.forward_pre(src)
        return self.forward_post(src)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
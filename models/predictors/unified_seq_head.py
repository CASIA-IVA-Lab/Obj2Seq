# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import math

from util.misc import inverse_sigmoid
from timm.models.layers import trunc_normal_
from .classifiers import build_label_classifier
from .seq_postprocess import build_sequence_postprocess
from ..transformer.attention_modules import DeformableDecoderLayer
from models.ops.functions import MSDeformAttnFunction
from models.losses.classwise_criterion import ClasswiseCriterion


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0., proj=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim) if proj else nn.Identity()

    def forward(self, x, pre_kv=None, attn_mask=None):
        N, B, C = x.shape
        qkv = self.qkv(x).reshape(N, B, 3, self.num_heads, C // self.num_heads).permute(2, 1, 3, 0, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if pre_kv is not None:
            k = torch.cat([pre_kv[0], k], dim=2)
            v = torch.cat([pre_kv[1], v], dim=2)
        pre_kv = torch.stack([k, v], dim=0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            attn.masked_fill_(attn_mask, float('-inf'))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(2, 0, 1, 3).reshape(N, B, C)
        x = self.proj(x)
        return x, pre_kv


def update_reference_points_xy(output_signals, reference_points, id_step):
    # reference_points: cs_all, nobj, 2
    # output_signals List( Tensor(cs_all, nobj) )
    if id_step < 2:
        new_reference_points = inverse_sigmoid(reference_points)
        new_reference_points[:, :, id_step] += output_signals[-1]
        new_reference_points = new_reference_points.sigmoid()
        return new_reference_points
    else:
        return reference_points


class UnifiedSeqHead(DeformableDecoderLayer):
    def __init__(self, args):
        # required keyts:
        #   num_steps (int), pos_emb (bool), sg_previous_logits (bool), combine_method (str)
        #   task_category (str: filename), args.num_classes (int)
        #   LOSS, CLASSIFIER
        #   other args as for decoder layer
        super().__init__(args)

        if args.no_ffn:
            del self.ffn
            self.ffn = nn.Identity()
        if self.self_attn:
            del self.self_attn
            self.self_attn = Attention(self.d_model, self.n_heads, dropout=args.dropout, proj=args.self_attn_proj)

        # TODO: Number of classes
        self.classifier = build_label_classifier(args.CLASSIFIER)
        self.num_steps = args.num_steps
        self.output_embeds = nn.ModuleList([
            MLP(self.d_model, self.d_model, c_out, 1) for c_out in [1] * self.num_steps
        ])
        self.reset_parameters_as_first_head()

        # TODO: more general functions
        self.adjust_reference_points = update_reference_points_xy
        if args.pos_emb:
            self.pos_emb = nn.Embedding(self.num_steps, self.d_model)
            trunc_normal_(self.pos_emb.weight, std=.02)
        else:
            self.pos_emb = None

        # for post logits
        self.post_process = build_sequence_postprocess(args)
        self.sg_previous_logits = args.sg_previous_logits
        self.combine_method = args.combine_method

        # for loss
        self.criterion = ClasswiseCriterion(args.LOSS)

    def reset_parameters_as_first_head(self):
        for i in range(self.num_steps):
            nn.init.constant_(self.output_embeds[i].layers[-1].weight.data, 0)
            nn.init.constant_(self.output_embeds[i].layers[-1].bias.data, 0. if (i < 2 or i >= 4) else -2.0)

    def reset_parameters_as_refine_head(self):
        for i in range(self.num_steps):
            nn.init.constant_(self.output_embeds[i].layers[-1].weight.data, 0)
            nn.init.constant_(self.output_embeds[i].layers[-1].bias.data, 0)

    def self_attn_forward(self, tgt, query_pos, **kwargs):
        # q = k = self.with_pos_embed(tgt, query_pos_self)
        bs, l, c = tgt.shape
        tgt2, self.pre_kv = self.self_attn(tgt.view(1, bs*l, c), pre_kv=self.pre_kv)
        return tgt2.view(bs, l, c)

    def forward(self, feat, query_pos, reference_points, srcs, src_padding_masks, **kwargs):
        # feat: cs_all, nobj, c
        # srcs: bs, l, c
        # reference_points: cs_all, nobj, 2
        cs_all, nobj, c = feat.shape

        # output for scores
        previous_logits = kwargs.pop("previous_logits", None)
        class_vector = kwargs.pop("class_vector", None)
        bs_idx, cls_idx = kwargs.pop("bs_idx", None), kwargs.pop("cls_idx", None)
        if kwargs.pop("rearrange", False):
            num_steps, cls_idx, feat, class_vector, bs_idx, kwargs["src_valid_ratios"] = self.post_process.taskCategory.arrangeBySteps(cls_idx, feat, class_vector, bs_idx, kwargs["src_valid_ratios"])
        else:
            num_steps = self.post_process.taskCategory.getNumSteps(cls_idx)
        output_classes = self.classifier(feat, class_vector=class_vector.unsqueeze(1) if class_vector is not None else None)
        output_classes = self.postprocess_logits(output_classes, previous_logits, bs_idx, cls_idx)

        # prepare for sequence
        input_feat = feat
        output_signals = [] # a list of Tensor(cs_all, nobj)
        original_reference_points = reference_points
        self.pre_kv = None
        self.cross_attn.preprocess_value(srcs, src_padding_masks, bs_idx=bs_idx)
        for id_step, output_embed in enumerate(self.output_embeds):
            # forward the features, get output_features
            if self.pos_emb is not None:
                feat = feat + self.pos_emb.weight[id_step]
            forward_reference_points = reference_points.detach()
            output_feat = super().forward(feat, query_pos, forward_reference_points, srcs, src_padding_masks, **kwargs)
            output_signal = output_embed(output_feat).squeeze(-1)
            output_signals.append(output_signal)

            feat = self.generate_feat_for_next_step(output_feat, output_signal, reference_points, None,id_step)
            reference_points = self.adjust_reference_points(output_signals, reference_points, id_step)
            # TODO: make this more suitable for other tasks
            if (num_steps == id_step + 1).sum() > 0 and id_step < self.num_steps:
                count_needs = (num_steps > id_step + 1).sum()
                old_cs = feat.shape[0]
                feat = feat[:count_needs]
                reference_points = reference_points[:count_needs]
                self.pre_kv = self.pre_kv.unflatten(1, (old_cs, nobj))[:, :count_needs].flatten(1,2)
                self.cross_attn.value = self.cross_attn.value[:count_needs]
                kwargs["src_valid_ratios"] = kwargs["src_valid_ratios"][:count_needs]

        outputs = self.post_process(output_signals, output_classes, original_reference_points, bs_idx, cls_idx)
        # prepare targets
        targets = kwargs.pop("targets", None)
        if targets is not None:
            loss_dict = self.criterion(outputs, targets)
        else:
            assert not self.training, "Targets are required for training mode (unified_seq_head.py)"
            loss_dict = {}
        return outputs, loss_dict

    def generate_feat_for_next_step(self, output_feat, output_signal, reference_logits, boxes, id_step):
        # prepare inputs for the next input
        # output_feat:   bs*cs*nobj, 1, c
        # output_signal: bs*cs*nobj, 1, 1
        # reference_points: bs*cs*nobj, 1, 2
        # boxes: bs*cs*nobj, 1, 4
        feat = output_feat.clone().detach()
        return feat

    def postprocess_logits(self, outputs_logits, previous_logits, bs_idx, cls_idx):
        if previous_logits is not None:
            previous_logits = previous_logits[bs_idx, cls_idx]
            previous_logits = previous_logits.unsqueeze(-1)
            if self.sg_previous_logits:
                previous_logits = previous_logits.detach()
        if self.combine_method =="none":
            return outputs_logits
        elif self.combine_method == "add":
            return outputs_logits.sigmoid() + previous_logits.sigmoid()
        elif self.combine_method == "multiple":
            return inverse_sigmoid(outputs_logits.sigmoid() * previous_logits.sigmoid())
        else:
            raise KeyError


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

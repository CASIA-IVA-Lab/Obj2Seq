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
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from util.misc import inverse_sigmoid

from .attention_modules import DeformableDecoderLayer, _get_clones
from ..predictors import build_detect_predictor

class ObjectDecoder(nn.Module):
    def __init__(self, d_model=256, args=None):
        super().__init__()
        self.d_model = d_model
        self.num_layers = args.num_layers
        self.num_position = args.num_query_position
        self.position_patterns = nn.Embedding(self.num_position, d_model)

        object_decoder_layer = DeformableDecoderLayer(args.LAYER)
        self.object_decoder_layers = _get_clones(object_decoder_layer, self.num_layers)

        # something else
        self._init_detect_head(args)
        self._init_reference_points(args)

    def _init_detect_head(self, args):
        self.detect_head = build_detect_predictor(args.HEAD)
        self.refine_reference_points = args.refine_reference_points

        if self.refine_reference_points:
            self.detect_head = _get_clones(self.detect_head, self.num_layers)
            # reset params
            for head in self.detect_head[1:]:
                head.reset_parameters_as_refine_head()
        else:
            self.detect_head = nn.ModuleList([self.detect_head for _ in range(self.num_layers)])

    def _init_reference_points(self, args):
        self.spatial_prior=args.spatial_prior
        if self.spatial_prior == "learned":
            self.position = nn.Embedding(self.num_position, 2)
            nn.init.uniform_(self.position.weight.data, 0, 1)
        elif self.spatial_prior == "sigmoid":
            self.position = nn.Embedding(self.num_position, self.d_model)
            self.reference_points = nn.Linear(self.d_model, 2)
            nn.init.xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            nn.init.constant_(self.reference_points.bias.data, 0.)
        self.with_query_pos_embed = args.with_query_pos_embed

    def forward(self, srcs, mask, targets=None, additional_info={}, kwargs={}):
        """
            srcs: bs, h, w, c || bs, l, c
            mask: bs, h, w || bs, l
            pos_emb:
        """
        class_vector = additional_info.pop("class_vector", None)
        previous_logits = additional_info.pop("previous_logits", None)

        bs = srcs.shape[0]
        bs_idx = additional_info.pop("bs_idx", torch.arange(bs))
        cls_idx = additional_info.pop("cls_idx", torch.zeros(bs, dtype=torch.long))

        # modify srcs
        cs_batch = [(bs_idx==i).sum().item() for i in range(bs)]
        cs_all = sum(cs_batch)
        # src is not modified, but in ops

        tgt_object = self.get_object_queries(class_vector, cls_idx, bs_idx)
        reference_points, query_pos_embed = self.get_reference_points(cs_all)

        # modify kwargs
        kwargs["src_valid_ratios"] = torch.cat([
            vs.expand(cs ,-1, -1) for cs, vs in zip(cs_batch, kwargs["src_valid_ratios"])
        ])
        kwargs["cs_batch"] = cs_batch

        all_outputs = []
        # prepare kwargs for predictor
        predictor_kwargs = {}
        for key in kwargs:
            predictor_kwargs[key] = kwargs[key]
        predictor_kwargs["class_vector"] = class_vector
        predictor_kwargs["bs_idx"] = bs_idx
        predictor_kwargs["cls_idx"] = cls_idx
        predictor_kwargs["previous_logits"] = previous_logits
        predictor_kwargs["targets"] = targets

        loss_dict = {}
        for lid, layer in enumerate(self.object_decoder_layers):
            tgt_object = layer(tgt_object, query_pos_embed, reference_points, srcs=srcs, src_padding_masks=mask, **kwargs)
            if self.training or self.refine_reference_points or lid == self.num_layers - 1:
                predictor_kwargs["rearrange"] = not self.refine_reference_points
                # TODO: move arrange into prompt indicator
                layer_outputs, layer_loss = self.detect_head[lid](tgt_object, query_pos_embed, reference_points, srcs=srcs, src_padding_masks=mask, **predictor_kwargs)
                if self.refine_reference_points:
                    reference_points = layer_outputs["detection"]["pred_boxes"].clone().detach()
                all_outputs.append(layer_outputs)
                for key in layer_loss:
                    loss_dict[f"{key}_{lid}"] = layer_loss[key]

        outputs = all_outputs.pop()
        return outputs, loss_dict

    def get_reference_points(self, bs):
        # Generate srcs
        if self.spatial_prior == "learned":
            reference_points = self.position.weight.unsqueeze(0).repeat(bs, 1, 1)
            query_embed = None
        elif self.spatial_prior == "grid":
            nx=ny=round(math.sqrt(self.num_position))
            self.num_position=nx*ny
            x = (torch.arange(nx) + 0.5) / nx
            y = (torch.arange(ny) + 0.5) / ny
            xy=torch.meshgrid(x,y)
            reference_points=torch.cat([xy[0].reshape(-1)[...,None],xy[1].reshape(-1)[...,None]],-1).cuda()
            reference_points = reference_points.unsqueeze(0).repeat(bs, 1, 1)
            query_embed = None
        elif self.spatial_prior == "sigmoid":
            query_embed = self.position.weight.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            if not self.with_query_pos_embed:
                query_embed = None
        else:
            raise ValueError(f'unknown {self.spatial_prior} spatial prior')
        return reference_points, query_embed

    def get_object_queries(self, class_vector, cls_idx, bs_idx):
        c = self.d_model
        if class_vector is None:
            cs_all = cls_idx.shape[0]
            tgt_object = self.position_patterns.weight.reshape(1 , self.num_position, c).repeat(cs_all, 1, 1)
        elif class_vector.dim() == 3:
            bs, cs, c = class_vector.shape
            tgt_pos = self.position_patterns.weight.reshape(1 , self.num_position, c).repeat(bs*cs, 1, 1)
            tgt_object = tgt_pos + class_vector.view(bs*cs, 1, c) # bs*cs, nobj, c
        elif class_vector.dim() == 2:
            cs_all, c = class_vector.shape
            tgt_pos = self.position_patterns.weight.reshape(1 , self.num_position, c).repeat(cs_all, 1, 1)
            tgt_object = tgt_pos + class_vector.view(cs_all, 1, c) # cs_all, nobj, c
        return tgt_object

# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
from torch import nn
from .attention_modules import DeformableEncoderLayer, _get_clones

class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        encoder_layer = DeformableEncoderLayer(args.ENCODER_LAYER)
        self.encoder_layers =  _get_clones(encoder_layer, args.enc_layers)
    
    def forward(self, tgt, *args, **kwargs):
        # tgt: bs, h, w, c || bs, l, c
        for layer in self.encoder_layers:
            tgt = layer(tgt, *args, **kwargs)
        return tgt


def build_encoder(args):
    return TransformerEncoder(args)
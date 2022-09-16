# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Swin Transformer (https://github.com/megvii-research/AnchorDETR)
# Copyright (c) 2021 Microsoft. Licensed under The MIT License.
# --------------------------------------------------------
# Modified from Anchor DETR (https://github.com/megvii-research/AnchorDETR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
from .resnet import build_backbone as build_resnet
from .swin_transformer import build_backbone as build_swin_transformer

def build_backbone(args):
    # include
    ## args.backbone
    if args.backbone.startswith("resnet"):
        args.defrost()
        args.RESNET.train_backbone = args.train_backbone
        args.RESNET.num_feature_levels = args.num_feature_levels
        args.freeze()
        return build_resnet(args.RESNET)
    elif "swin" in args.backbone:
        args.defrost()
        args.SWIN.train_backbone = args.train_backbone
        args.SWIN.num_feature_levels = args.num_feature_levels
        args.freeze()
        return build_swin_transformer(args.SWIN)
    else:
        raise NotImplemented
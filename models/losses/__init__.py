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
from .losses import sigmoid_focal_loss
from .asl_losses import AsymmetricLoss, AsymmetricLossOptimized


def build_asymmetricloss(args):
    lossClass = AsymmetricLossOptimized if args.asl_optimized else AsymmetricLoss
    return lossClass(gamma_neg=args.asl_gamma_neg,
                     gamma_pos=args.asl_gamma_pos,
                     clip=args.asl_clip,
                     disable_torch_grad_focal_loss=True)

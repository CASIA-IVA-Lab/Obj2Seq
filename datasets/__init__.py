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
import torch.utils.data
from .torchvision_datasets import CocoDetection
from .dataloader import build_dataloader

from .transform_pipelines import build_transform
from .coco import build as build_coco
from .coco_hybrid import build as build_coco_hybrid
from .coco_mtl import build_coco_mtl
from .imnet import build_dataset as build_imnet


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):
    transform = build_transform(image_set, args.DATA.TRANSFORM)
    if args.DATA.type == 'coco':
        return build_coco(image_set, transform, args.DATA)
    if args.DATA.type == 'coco_hybrid':
        return build_coco_hybrid(image_set, transform, args.DATA)
    if args.DATA.type == 'coco_mtl':
        return build_coco_mtl(image_set, transform, args.DATA)
    if args.DATA.type == 'imnet':
        return build_imnet(image_set, transform, args.DATA)
    if args.DATA.type == 'coco_panoptic':
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic
        return build_coco_panoptic(image_set, args)
    raise ValueError(f'DATA type {args.DATA.type} not supported')

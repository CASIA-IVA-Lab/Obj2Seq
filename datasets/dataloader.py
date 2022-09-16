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
from functools import partial

import torch
from torch.utils.data import DataLoader

import datasets.samplers as samplers
from util.collate_fn import build_collate_fn


class HybridBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, *args, **kwargs):
        self.kps_sampler = kwargs.pop("kps_sampler")
        self.kps_iter = iter(self.kps_sampler)
        super().__init__(*args, **kwargs)
        assert self.batch_size == 1

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            try:
                idx_kps = next(self.kps_iter)
            except StopIteration:
                self.kps_sampler.set_epoch(self.kps_sampler.epoch * 3)
                self.kps_iter = iter(self.kps_sampler)
                idx_kps = next(self.kps_iter)
            batch.append(-idx_kps)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch


def build_dataloader(dataset_train, dataset_val, args):
    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train, fix_split = args.sampler_fix_split)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if args.type == "coco_hybrid":
        kps_sampler_train = samplers.DistributedSampler(dataset_train.dset_kps)
        batch_sampler_train = HybridBatchSampler(
            sampler_train, args.batch_size, drop_last=True, kps_sampler=kps_sampler_train)
        output_sampler = [sampler_train, kps_sampler_train]
    else:
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)
        output_sampler = sampler_train

    collate_fn = build_collate_fn(args.COLLECT_FN)
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)
    return data_loader_train, data_loader_val, output_sampler
# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
from .misc import nested_tensor_from_tensor_list


class BaseCollator():
    def __init__(self, fix_input=None, input_divisor=None):
        self.fix_input = fix_input
        self.input_divisor = input_divisor
        assert self.fix_input is None or self.input_divisor is None

    def __call__(self, batch):
        batch = list(zip(*batch))
        batch[0] = nested_tensor_from_tensor_list(batch[0], fix_input=self.fix_input, input_divisor=self.input_divisor)
        return tuple(batch)


class CLSCollator():
    def __init__(self, fix_input=None, input_divisor=None):
        self.fix_input = fix_input
        self.input_divisor = input_divisor
        assert self.fix_input is None or self.input_divisor is None

    def __call__(self, batch):
        batch = list(zip(*batch))
        batch[0] = nested_tensor_from_tensor_list(batch[0], fix_input=self.fix_input, input_divisor=self.input_divisor)
        if "boxes_by_cls" in batch[1][0]:
            # process targets
            overall_batch = dict()
            overall_batch["image_id"] = torch.stack([item["image_id"] for item in batch[1]])
            overall_batch["size"] = torch.stack([item["size"] for item in batch[1]])
            overall_batch["orig_size"] = torch.stack([item["orig_size"] for item in batch[1]])
            overall_batch["class_label"] = [item["labels"].unique() for item in batch[1]]
            overall_batch["multi_label"] = [item["multi_label"] for item in batch[1]]
            overall_batch["super_label"] = [item["super_label"] for item in batch[1]]
            overall_batch["boxes_by_cls"] = [item["boxes_by_cls"] for item in batch[1]]
            if "keypoints_by_cls" in batch[1][0]:
                overall_batch["keypoints_by_cls"] = [item["keypoints_by_cls"] for item in batch[1]]
            if "object_class" in batch[1][0]:
                overall_batch["object_class"] = torch.stack([item["object_class"] for item in batch[1]])
            batch[1] = overall_batch
        return tuple(batch)


def collate_fn_imnet(batch, is_train=True):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    if is_train:
        batch[1] = {"multi_label": [torch.as_tensor([item]) for item in batch[1]]}
    else:
        batch[1] = [{"multi_label": torch.as_tensor([item])} for item in batch[1]]
    return tuple(batch)


def build_collate_fn(args):
    if args.type == "COCObyCLS":
        return CLSCollator(args.fix_input, args.input_divisor)
    elif args.type == "none":
        return BaseCollator(args.fix_input, args.input_divisor)

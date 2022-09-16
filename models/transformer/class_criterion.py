# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.losses import build_asymmetricloss
from util.misc import get_world_size, is_dist_avail_and_initialized


class ClassDecoderCriterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.losses = args.losses
        self.loss_weights = {
            "asl": args.asl_loss_weight,
            "bce": args.asl_loss_weight,
        }
        self.asl_loss = build_asymmetricloss(args)
        self.loss_funcs = {
            "asl": lambda outputs, targets: self.asl_loss(outputs['cls_label_logits'], targets["multi_label_onehot"], targets["multi_label_weights"]),
            "bce": lambda outputs, targets: F.binary_cross_entropy_with_logits(outputs['cls_label_logits'], targets["multi_label_onehot"], targets["multi_label_weights"], reduction="sum") / targets["multi_label_weights"].sum(),
        }

    def prepare_targets(self, outputs, targets):
        return {
            "multi_label_onehot": torch.stack([t["multi_label_onehot"] for t in targets], dim=0),
            "multi_label_weights": torch.stack([t["multi_label_weights"] for t in targets], dim=0),
        }

    def forward(self, outputs, aux_outputs, targets):
        targets = self.prepare_targets(outputs, targets)
        loss_dict = {}
        for loss in self.losses:
            loss_dict[f"cls_{loss}"] = self.loss_weights[loss] * self.loss_funcs[loss](outputs, targets)
            for i, aux_label_output in enumerate(aux_outputs):
                loss_dict[f"cls_{loss}_{i}"] = self.loss_weights[loss] * self.loss_funcs[loss](aux_label_output, targets)
        return loss_dict
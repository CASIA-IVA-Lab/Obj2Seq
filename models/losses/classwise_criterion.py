# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
import torch
from torch import nn

from util.misc import (nested_tensor_from_tensor_list, interpolate,
                       is_main_process, get_world_size, is_dist_avail_and_initialized)
from util.task_category import TaskCategory

from .unified_single_class_criterion import UnifiedSingleClassCriterion


class ClasswiseCriterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.set_criterion = UnifiedSingleClassCriterion(args)
        self.taskCategory = TaskCategory(args.task_category, args.num_classes)
        self.need_keypoints = "pose" in [it.name for it in self.taskCategory.tasks]

    def forward(self, outputs, targets):
        # outputs: Dict{}
        ##  output_per_task: Dict{}
        ###   batch_index: cs_all
        ###   class_index: cs_all
        ###   pred_logits: cs_all, nobj
        ###   pred_boxes:  cs_all, nobj, 4
        # targets:
        loss_dicts_all = []
        for tKey, output in outputs.items():
            device = output['pred_logits'].device
            cs_all, num_obj = output['pred_logits'].shape
            num_boxes = self.get_num_boxes(targets, device)
            num_pts = self.get_num_pts(targets, device) if self.need_keypoints else 1
            num_people = self.get_num_people(targets, device) if self.need_keypoints else 1
            bs_idx, cls_idx = output["batch_index"], output["class_index"] # cs_all

            task_info = self.taskCategory[tKey]
            target = []
            for id_b, id_c in zip(bs_idx, cls_idx):
                tgtThis = {}
                id_c = id_c.item()
                if id_c in targets[id_b]:
                    tgtOrigin = targets[id_b][id_c]
                    for key in task_info.required_targets:
                        tgtThis[key] = tgtOrigin[key]
                else:
                    for key in task_info.required_targets:
                        default_shape = task_info.required_targets[key]
                        tgtThis[key] = torch.zeros(default_shape, device=device)
                target.append(tgtThis)

            # TODO: form class of the same task into a batch
            loss_dicts_all.append(self.set_criterion(output, target, task_info.losses, num_boxes, num_pts, num_people))

        loss_dict = {}
        for idict in loss_dicts_all:
            for k in idict:
                if k in loss_dict:
                    loss_dict[k] += idict[k]
                else:
                    loss_dict[k] = idict[k]
        return loss_dict

    def get_num_boxes(self, targets, device):
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(sum(t[key]['boxes'].shape[0] for key in t if isinstance(key, int)) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_num_pts(self, targets, device):
        kps = [t[0]["keypoints"] for t in targets if 0 in t]
        if len(kps) > 0:
            kps = torch.cat(kps, dim=0)
            kps = (kps[..., 2] > 0) * (kps[..., :2] >= 0).all(dim=-1) * (kps[..., :2] <= 1).all(dim=-1)
            num_pts = kps.sum()
        else:
            num_pts = torch.as_tensor(0., device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_pts)
        num_pts = torch.clamp(num_pts / get_world_size(), min=1).item()
        return num_pts

    def get_num_people(self, targets, device):
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_people = sum(t[0]['boxes'].shape[0] for t in targets if 0 in t)
        num_people = torch.as_tensor([num_people], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_people)
        num_people = torch.clamp(num_people / get_world_size(), min=1).item()
        return num_people

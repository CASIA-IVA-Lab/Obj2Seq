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
import torch
import torch.nn.functional as F
from torch import nn

from .unified_matcher import build_matcher
from .losses import sigmoid_focal_loss
from util import box_ops
from util.misc import (nested_tensor_from_tensor_list, interpolate,
                       get_world_size, is_dist_avail_and_initialized)


import numpy as np
KPS_OKS_SIGMAS = np.array([
    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
    .87, .87, .89, .89
]) / 10.0


class UnifiedSingleClassCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, args):
        """ Create the criterion.
        Parameters:
            args.MATCHER: module able to compute a matching between targets and proposals
            args.focal_alpha: dict containing as key the names of the losses and as values their relative weight.
            args.*_loss_coef
            args.*_normalization
            args.bce_negative_weight
        """
        super().__init__()
        self.matcher = build_matcher(args.MATCHER)
        self.focal_alpha = args.focal_alpha
        # weight dict
        all_weight_dict = {
            'loss_ce': args.cls_loss_coef,
            'loss_bce': args.cls_loss_coef,
            'loss_bbox': args.bbox_loss_coef,
            'loss_giou': args.giou_loss_coef,
            'loss_kps_l1': args.keypoint_l1_loss_coef,
            'loss_oks': args.keypoint_oks_loss_coef,
        }
        self.all_weight_dict = all_weight_dict
        # loss weight
        self.bce_negative_weight = args.bce_negative_weight
        self.class_normalization = args.class_normalization
        self.box_normalization = "num_box"
        self.keypoint_normalization = args.keypoint_normalization
        self.oks_normalization = args.oks_normalization

    def loss_labels(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1]],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot[idx] = 1

        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes=None, alpha=self.focal_alpha, gamma=2) / self.loss_normalization[self.class_normalization]
        losses = {'loss_ce': loss_ce}

        if False:#log:
            # TODO this should probably be a separate loss, not hacked in this one here
            # TODO Fix here
            losses['class_error'] = 100 # - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_bce(self, outputs, targets, indices):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # assert self.type == 'det'
        srcs_idx = self._get_src_permutation_idx(indices)

        src_logits = outputs['pred_logits'].sigmoid()
        # valid_src_flag = torch.ones_like(src_logits)
        # valid_src_flag[srcs_idx] = targets['with_joint_flag'][tgts_idx].float()
        target_logits = torch.zeros_like(src_logits)
        target_logits[srcs_idx] = 1.0
        weight_matrix = torch.full_like(src_logits, self.bce_negative_weight)
        weight_matrix[srcs_idx] = 1.0

        loss_bce = F.binary_cross_entropy(src_logits, target_logits, weight=weight_matrix, reduction='sum')
        loss_bce = loss_bce / self.loss_normalization[self.class_normalization]
        # loss_bce = loss_bce * valid_src_flag
        # loss_bce = loss_bce.sum() / valid_src_flag.sum()
        losses = {'loss_bce': loss_bce}

        return losses

    def loss_boxes(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / self.loss_normalization[self.box_normalization]

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / self.loss_normalization[self.box_normalization]
        return losses

    def loss_oks(self, outputs, targets, indices, with_center=True, eps=1e-15):

        idx = self._get_src_permutation_idx(indices)
        src_joints = outputs['pred_keypoints'][idx] # tgt, 17, 2
        tgt_joints = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0) # tgt, 17, 3
        tgt_bboxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0) # tgt, 4

        tgt_flags = tgt_joints[..., 2]
        tgt_joints = tgt_joints[..., 0:2]
        tgt_flags = (tgt_flags > 0) * (tgt_joints >= 0).all(dim=-1) * (tgt_joints <= 1).all(dim=-1) # zychen
        tgt_wh = tgt_bboxes[..., 2:]
        tgt_areas = tgt_wh[..., 0] * tgt_wh[..., 1]
        sigmas = KPS_OKS_SIGMAS # self.sigmas

        # if with_center:
        #     tgt_center = tgt_bboxes[..., 0:2]
        #     sigma_center = sigmas.mean()
        #     tgt_joints = torch.cat([tgt_joints, tgt_center[:, None, :]], dim=1)
        #     sigmas = np.append(sigmas, np.array([sigma_center]), axis=0)
        #     tgt_flags = torch.cat([tgt_flags, torch.ones([tgt_flags.size(0), 1]).type_as(tgt_flags)], dim=1)

        sigmas = torch.tensor(sigmas).type_as(tgt_joints)
        d_sq = torch.square(src_joints - tgt_joints).sum(-1)
        loss_oks = 1 - torch.exp(-1 * d_sq / (2 * tgt_areas[:, None] * sigmas[None, :] + 1e-15))
        # loss_oks = loss_oks * tgt_flags * with_joint_flag[:, None]
        loss_oks = loss_oks * tgt_flags
        loss_oks = loss_oks.sum(-1) / (tgt_flags.sum(-1) + eps) # 每一个object单独计算了oks
        loss_oks = loss_oks.sum() / (self.loss_normalization[self.oks_normalization] + eps)
        losses = {'loss_oks': loss_oks}

        return losses

    def loss_keypoints(self, outputs, targets, indices): #TODO: num_pts
        assert 'pred_keypoints' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_kps = outputs['pred_keypoints'][idx]
        target_kps = torch.cat([t['keypoints'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        tgt_kps = target_kps[..., :2]
        tgt_visible = (target_kps[..., 2] > 0) * (tgt_kps >= 0).all(dim=-1) * (tgt_kps <= 1).all(dim=-1)
        src_loss, tgt_loss = src_kps[tgt_visible], tgt_kps[tgt_visible]

        loss_keypoint = F.l1_loss(src_loss, tgt_loss, reduction="sum")
        loss_keypoint = loss_keypoint / self.loss_normalization[self.keypoint_normalization]
        return {"loss_kps_l1": loss_keypoint}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices):
        loss_map = {
            'labels': self.loss_labels,
            'bce': self.loss_bce,
            'boxes': self.loss_boxes,
            'keypoints': self.loss_keypoints,
            'oks': self.loss_oks,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices)

    def build_weight_dict(self, losses):
        weight_dict = {}
        if "labels" in losses:
            weight_dict["loss_ce"] = self.all_weight_dict["loss_ce"]
        if "bce" in losses:
            weight_dict["loss_bce"] = self.all_weight_dict["loss_bce"]
        if "boxes" in losses:
            weight_dict["loss_bbox"] = self.all_weight_dict["loss_bbox"]
            weight_dict["loss_giou"] = self.all_weight_dict["loss_giou"]
        if "keypoints" in losses:
            weight_dict['loss_kps_l1'] = self.all_weight_dict['loss_kps_l1']
        if "oks" in losses:
            weight_dict['loss_oks'] = self.all_weight_dict['loss_oks']
        return weight_dict

    def forward(self, outputs, targets, losses, num_boxes, num_pts, num_people):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
                  pred_logits: bs, nobj
                  pred_boxes:  bs, nobj, 4
                  (optional):  bs, nobj, mngts
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
                  keypoints: ngts, 17, 3
        """

        weight_dict = self.build_weight_dict(losses)
        # normalize term
        self.loss_normalization = {"num_box": num_boxes, "num_pts": num_pts, "num_people": num_people, "mean": outputs["pred_logits"].shape[1], "none": 1}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs, targets, weight_dict, num_boxes, num_pts, num_people)

        # Compute all the requested losses
        loss_dict = {}
        for loss in losses:
            loss_dict.update(self.get_loss(loss, outputs, targets, indices))

        return self.rescale_loss(loss_dict, weight_dict)

    def rescale_loss(self, loss_dict, weight_dict):
        return {
            k: loss_dict[k] * weight_dict[k]
            for k in loss_dict.keys() if k in weight_dict
        }

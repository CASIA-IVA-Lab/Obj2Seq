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
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

import numpy as np
from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


KPS_OKS_SIGMAS = np.array([
    .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07,
    .87, .87, .89, .89
]) / 10.0


def joint_oks(src_joints, tgt_joints, tgt_bboxes, joint_sigmas=KPS_OKS_SIGMAS, with_center=True, eps=1e-15):
    tgt_flags = tgt_joints[:, :, 2]
    tgt_joints = tgt_joints[:, :, 0:2]
    tgt_wh = tgt_bboxes[..., 2:]
    tgt_areas = tgt_wh[..., 0] * tgt_wh[..., 1]
    num_gts, num_kpts = tgt_joints.shape[0:2]

    # if with_center:
    #     assert src_joints.size(1) == tgt_joints.size(1) + 1
    #     tgt_center = tgt_bboxes[..., 0:2]
    #     sigma_center = joint_sigmas.mean()
    #     tgt_joints = torch.cat([tgt_joints, tgt_center[:, None, :]], dim=1)
    #     joint_sigmas = np.append(joint_sigmas, np.array([sigma_center]), axis=0)
    #     tgt_flags = torch.cat([tgt_flags, torch.ones([num_gts, 1]).type_as(tgt_flags)], dim=1)
    #     num_kpts = num_kpts + 1

    areas = tgt_areas.unsqueeze(1).expand(num_gts, num_kpts)
    sigmas = torch.tensor(joint_sigmas).type_as(tgt_joints)
    sigmas_sq = torch.square(2 * sigmas).unsqueeze(0).expand(num_gts, num_kpts)
    d_sq = torch.square(src_joints.unsqueeze(1) - tgt_joints.unsqueeze(0)).sum(-1)
    tgt_flags = tgt_flags.unsqueeze(0).expand(*d_sq.shape)

    oks = torch.exp(-d_sq / (2 * areas * sigmas_sq + eps))
    oks = oks * tgt_flags
    oks = oks.sum(-1) / (tgt_flags.sum(-1) + eps)

    return oks


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, args):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.class_normalization = args.set_class_normalization
        self.box_normalization = args.set_box_normalization
        self.keypoint_normalization = args.set_keypoint_normalization
        self.oks_normalization = args.set_oks_normalization

    def forward(self, outputs, targets, weight_dict, num_box, num_pts, num_people):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

            match_args

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]
            # how to normalize loss for both class, box and keypoint
            NORMALIZER = {"num_box": num_box, "num_pts": num_pts, "num_people": num_people, "mean": num_queries, "none": 1, "box_average": num_box}
            with_boxes = True
            with_keypoints = "loss_kps_l1" in weight_dict or "loss_oks" in weight_dict

            # We flatten to compute the cost matrices in a batch
            out_logit = outputs["pred_logits"].flatten(0, 1) # [batch_size * num_queries]
            out_prob = out_logit.sigmoid()
            if with_boxes:
                out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]
            if with_keypoints:
                out_keypoints = outputs["pred_keypoints"].flatten(0, 1)

            # Also concat the target labels and boxes
            # tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])
            sizes = [t["boxes"].shape[0] for t in targets]
            num_local = sum(sizes)

            if num_local == 0:
                return [(torch.as_tensor([], dtype=torch.int64), torch.as_tensor([], dtype=torch.int64)) for _ in sizes]

            assert ("loss_bce" in weight_dict) ^ ("loss_ce" in weight_dict)
            # Compute the classification cost.
            if "loss_bce" in weight_dict:
                cost_class = - out_prob * weight_dict["loss_bce"]
            elif "loss_ce" in weight_dict:
                alpha = 0.25
                gamma = 2.0
                neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
                pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
                cost_class = pos_cost_class - neg_cost_class # [batch_size * num_queries]
                cost_class = cost_class * weight_dict["loss_ce"]
            cost_class = cost_class[..., None].repeat(1, num_local)

            C = cost_class / NORMALIZER[self.class_normalization]

            if with_boxes:
                # Compute the L1 cost between boxes
                cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) / NORMALIZER[self.box_normalization]

                # Compute the giou cost betwen boxes
                cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox),
                                                box_cxcywh_to_xyxy(tgt_bbox)) / NORMALIZER[self.box_normalization]
                # Final cost matrix
                C_box = weight_dict["loss_bbox"] * cost_bbox + weight_dict["loss_giou"] * cost_giou
                C = C + C_box

            if with_keypoints:
                tgt_kps = torch.cat([v["keypoints"] for v in targets])# tgt, 17, 3
                tgt_visible = tgt_kps[..., -1] # tgt, 17
                tgt_kps = tgt_kps[..., :2] # tgt, 17, 2
                tgt_visible = (tgt_visible > 0) * (tgt_kps >= 0).all(dim=-1) * (tgt_kps <= 1).all(dim=-1) # # tgt, 17
                if "loss_kps_l1" in weight_dict:
                    out_kps = out_keypoints.unsqueeze(1) # bs*nobj, 1, 17, 2
                    tgt_kps_t = tgt_kps.unsqueeze(0) # 1, tgt, 17, 2

                    cost_kps_l1 = torch.abs(out_kps - tgt_kps_t).sum(-1) * tgt_visible # # bs*nobj, tgt, 17
                    # cost_kps_l1 = torch.cdist(out_kps, tgt_kps_t, p=1).permute(1, 2, 0) * tgt_visible # bs*nobj, tgt, 17
                    cost_kps_l1 = cost_kps_l1.sum(-1)
                    if self.keypoint_normalization == "box_average":
                        cost_kps_l1 = cost_kps_l1 / tgt_visible.sum(-1).clamp(min=1.)
                    C_kps_l1 = weight_dict["loss_kps_l1"] * cost_kps_l1 / NORMALIZER[self.keypoint_normalization]
                    C = C + C_kps_l1

                if "loss_oks" in weight_dict:
                    # Compute the relative oks cost between joints
                    cat_tgt_kps = torch.cat([tgt_kps, tgt_visible.unsqueeze(-1)], dim=-1)
                    cost_oks = -joint_oks(out_keypoints, cat_tgt_kps, tgt_bbox)
                    C_kps_oks = weight_dict["loss_oks"] * cost_oks / NORMALIZER[self.oks_normalization]
                    C = C + C_kps_oks

            C = C.view(bs, num_queries, -1).cpu()

            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(args)

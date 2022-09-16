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
PostProcessor for Obj2Seq
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MutiClassPostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1) # bs, 4

        if "detection" in outputs:
            output = outputs["detection"]
            bs_idx, cls_idx = output["batch_index"], output["class_index"] # cs_all
            box_scale = scale_fct[bs_idx] # cs_all, 4
            all_scores = output["pred_logits"].sigmoid()
            nobj = all_scores.shape[-1]
            all_boxes = box_ops.box_cxcywh_to_xyxy(output["pred_boxes"]) * box_scale[:, None, :]
            results_det = []
            for id_b in bs_idx.unique():
                out_scores = all_scores[bs_idx == id_b].flatten() # cs_all*nobj
                out_bbox = all_boxes[bs_idx == id_b].flatten(0, 1)
                out_labels = output['class_index'][bs_idx == id_b].unsqueeze(-1).expand(-1, nobj).flatten() # cs_all*nobj

                s, indices = out_scores.sort(descending=True)
                s, indices = s[:100], indices[:100]
                results_det.append({'scores': s, 'labels': out_labels[indices], 'boxes': out_bbox[indices, :]})
            return results_det

        if "pose" in outputs:
            output = outputs["pose"]
            bs_idx, cls_idx = output["batch_index"], output["class_index"] # cs_all
            box_scale = scale_fct[bs_idx] # cs_all, 4
            all_scores = output["pred_logits"].sigmoid()
            nobj = all_scores.shape[-1]
            all_keypoints = output["pred_keypoints"] * box_scale[:, None, None, :2]
            all_keypoints = torch.cat([all_keypoints, torch.ones_like(all_keypoints)[..., :1]], dim=-1)
            all_boxes = box_ops.box_cxcywh_to_xyxy(output["pred_boxes"]) * box_scale[:, None, :]
            results_det = []
            for id_b in bs_idx.unique():
                out_scores = all_scores[bs_idx == id_b].flatten() # cs_all*nobj
                out_bbox = all_boxes[bs_idx == id_b].flatten(0, 1)
                out_keypoints = all_keypoints[bs_idx == id_b].flatten(0, 1)
                out_labels = torch.zeros_like(out_scores, dtype=torch.long)

                s, indices = out_scores.sort(descending=True)
                s, indices = s[:100], indices[:100]
                results_det.append({'scores': s, 'labels': out_labels[indices], 'boxes': out_bbox[indices], 'keypoints': out_keypoints[indices, :]})
            return results_det


class KeypointPostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs(Dict):
                pred_logits: Tensor [bs, nobj] (Currently support [bs, 1, nobj] too, may deprecated Later)
                pred_keypoints: Tensor [bs, nobj, 17, 2] (Currently support "keypoint_offsets" too)
                pred_boxes: Tensor [bs, nobj, 4] (Currently support [bs, 1, nobj, 4] too, may deprecated Later)
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        # squeeze operation is for old version
        if "pred_keypoints" in outputs:
            out_logits, out_keypoints = outputs['pred_logits'].squeeze(1), outputs["pred_keypoints"]
        else:
            # also for adaption
            out_logits, out_keypoints = outputs['pred_logits'].squeeze(1), outputs["keypoint_offsets"]
        bs, num_obj = out_logits.shape

        scores = out_logits.sigmoid() # bs, obj
        labels = torch.zeros_like(scores, dtype=torch.int64)

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1) # bs
        scale_fct = torch.stack([img_w, img_h], dim=1) # bs, 2
        out_keypoints = out_keypoints * scale_fct[:, None, None, :] # bs, nobj, 17, 2
        ones = torch.ones_like(out_keypoints)[..., :1]
        keypoints = torch.cat([out_keypoints, ones], dim=-1)

        if "pred_boxes" in outputs:
            out_bbox =  outputs['pred_boxes'].squeeze(1)
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox) # b, obj, 4
            scale_fct = torch.cat([scale_fct, scale_fct], dim=1) # bs, 4
            boxes = boxes * scale_fct[:, None, :]
        else:
            boxes = None

        results = []
        for idb, (s, l, k) in enumerate(zip(scores, labels, keypoints)):
            s, indices = s.sort(descending=True)
            s, indices = s[:100], indices[:100]
            #num_s = (s > 0.05).sum()
            #s, indices = s[:num_s], indices[:num_s]
            results.append({'scores': s, 'labels': l[indices], 'keypoints': k[indices, :]})
            # add box if possible
            if boxes is not None:
                results[-1]['boxes'] = boxes[idb, indices, :]

        return results


def build_postprocessor(args):
    if args.EVAL.postprocessor == "MultiClass":
        postprocessor = MutiClassPostProcess()
    elif args.EVAL.postprocessor == "Detr":
        postprocessor = PostProcess()
    elif args.EVAL.postprocessor == "Keypoint":
        postprocessor = KeypointPostProcess()
    else:
        raise KeyError
    return postprocessor
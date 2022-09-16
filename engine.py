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
Train and eval functions used in main.py
"""
import math
import os
import sys
import numpy as np
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from datasets.voc_map import voc_mAP, topk_recall, thr_recall


def convert_to_device(src, device):
    if isinstance(src, list):
        return [convert_to_device(item, device) for item in src]
    elif isinstance(src, dict):
        return {k: convert_to_device(v, device) for k, v in src.items()}
    elif isinstance(src, torch.Tensor):
        return src.to(device)


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, preprocessor=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets = prefetcher.next()

    # for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
    for _ in metric_logger.log_every(range(len(data_loader)), print_freq, header):
        if preprocessor is not None:
            samples, targets = preprocessor(samples, targets)

        outputs, loss_dict = model(samples, targets)
        losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced = {k: v
                             for k, v in loss_dict_reduced.items()}
        losses_reduced = sum(loss_dict_reduced.values())

        det_loss  = sum(loss_dict_reduced[k] for k in loss_dict_reduced.keys() if 'kps' not in k).item()
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, det_loss=det_loss, **loss_dict_reduced)
        # metric_logger.update(loss=loss_value)
        # metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        samples, targets = prefetcher.next()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_coco(model, postprocessor, data_loader, base_ds, device, output_dir, args=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = args.iou_types
    coco_evaluator = CocoEvaluator(base_ds, iou_types, save_json=args.save_json)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    saved_data = []
    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples.to(device)
        targets = convert_to_device(targets, device)

        outputs, loss_dict = model(samples, targets)
        if False:#"cls_label_logits" in outputs:
            # save some data
            label_logits = outputs["cls_label_logits"]
            label_probs = label_logits.sigmoid()
            # target_vector
            target_onehot = torch.zeros(label_probs.shape)
            for i_target, tgt in enumerate(targets):
                target_onehot[i_target, tgt["class_label"]] = 1

            _item = torch.cat((label_probs.detach().cpu(), target_onehot), 1)
            saved_data.append(_item)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced = {k: v
                             for k, v in loss_dict_reduced.items()}
        losses_reduced = sum(loss_dict_reduced.values()).item()
        det_loss  = sum(loss_dict_reduced[k] for k in loss_dict_reduced.keys() if 'kps' not in k).item()
        metric_logger.update(loss=losses_reduced, det_loss=det_loss, **loss_dict_reduced)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessor(outputs, orig_target_sizes)
        res = {tgt["image_id"].item(): output for tgt, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # save
    if False:
        mtl_dir = os.path.join(output_dir, "mtl")
        indices_dict = model.module.transformer.object_decoder.detect_head[-1].criterion.set_criterion.indices_dict
        torch.save(indices_dict, os.path.join(mtl_dir, f'saved_dict_{utils.get_rank()}.pth'))

    # calculate mAP
    if len(saved_data) > 0:
        mtl_dir = os.path.join(output_dir, "mtl")
        if not os.path.exists(mtl_dir):
            os.mkdir(mtl_dir)
        saved_data = torch.cat(saved_data, 0).numpy()
        saved_name = 'saved_data_tmp.{}.txt'.format(utils.get_rank())
        np.savetxt(os.path.join(mtl_dir, saved_name), saved_data)
        if utils.get_world_size() > 1:
            torch.distributed.barrier()

        if utils.get_rank() == 0:
            print("Calculating MTL mAP:")
            num_classes = saved_data.shape[1] // 2
            filenamelist = ['saved_data_tmp.{}.txt'.format(ii) for ii in range(utils.get_world_size())]
            mAP, aps = voc_mAP([os.path.join(mtl_dir, _filename) for _filename in filenamelist], num_classes, return_each=True)
            print("  mAP: {}".format(mAP))
            print("   aps: {}".format(np.array2string(aps, precision=5)))
            print("Calculating MTL Recall with TopK:")
            recall_results = topk_recall([os.path.join(mtl_dir, _filename) for _filename in filenamelist], num_classes, ks=[20,30,50,num_classes])
            for k, v in recall_results.items():
                print("   Recall@{}: {}".format(k, v))
            recall_results, recall_counts = thr_recall([os.path.join(mtl_dir, _filename) for _filename in filenamelist], num_classes, ks=[0.5, 0.3, 0.1, 0.01])
            print("Calculating MTL Recall with Thrs:")
            for k in recall_results:
                print("   Counts@{}: {}".format(k, recall_counts[k]))
                print("   Recall@{}: {}".format(k, recall_results[k]))
        else:
            mAP = 0

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    if args.save_json:
        for iou_type in iou_types:
            all_res = utils.all_gather(coco_evaluator.results[iou_type])
            results=[]
            for p in all_res:
                results.extend(p)
            coco_evaluator.results[iou_type] = results

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        for iou_type in iou_types:
            stats[f'coco_eval_{iou_type}'] = coco_evaluator.coco_eval[iou_type].stats.tolist()
    return stats, coco_evaluator



@torch.no_grad()
def evaluate_coco_mtl(model, postprocessor, data_loader, base_ds, device, output_dir, args=None):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    saved_data = []
    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples.to(device)
        targets = convert_to_device(targets, device)

        outputs, loss_dict = model(samples, targets)
        if "cls_label_logits" in outputs:
            # save some data
            label_logits = outputs["cls_label_logits"]
            label_probs = label_logits.sigmoid()
            # target_vector
            target_onehot = targets.cpu()

            _item = torch.cat((label_probs.detach().cpu(), target_onehot), 1)
            saved_data.append(_item)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced = {k: v
                             for k, v in loss_dict_reduced.items()}
        losses_reduced = sum(loss_dict_reduced.values()).item()
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)

    # calculate mAP
    if len(saved_data) > 0:
        mtl_dir = os.path.join(output_dir, "mtl")
        if not os.path.exists(mtl_dir):
            os.mkdir(mtl_dir)
        saved_data = torch.cat(saved_data, 0).numpy()
        saved_name = 'saved_data_tmp.{}.txt'.format(utils.get_rank())
        np.savetxt(os.path.join(mtl_dir, saved_name), saved_data)
        if utils.get_world_size() > 1:
            torch.distributed.barrier()

        if utils.get_rank() == 0:
            print("Calculating MTL mAP:")
            num_classes = saved_data.shape[1] // 2
            filenamelist = ['saved_data_tmp.{}.txt'.format(ii) for ii in range(utils.get_world_size())]
            mAP, aps = voc_mAP([os.path.join(mtl_dir, _filename) for _filename in filenamelist], num_classes, return_each=True)
            print("  mAP: {}".format(mAP))
            print("   aps: {}".format(np.array2string(aps, precision=5)))
            print("Calculating MTL Recall with TopK:")
            recall_results = topk_recall([os.path.join(mtl_dir, _filename) for _filename in filenamelist], num_classes, ks=[20,30,50,num_classes])
            for k, v in recall_results.items():
                print("   Recall@{}: {}".format(k, v))
            recall_results, recall_counts = thr_recall([os.path.join(mtl_dir, _filename) for _filename in filenamelist], num_classes, ks=[0.5, 0.3, 0.1, 0.01])
            print("Calculating MTL Recall with Thrs:")
            for k in recall_results:
                print("   Counts@{}: {}".format(k, recall_counts[k]))
                print("   Recall@{}: {}".format(k, recall_results[k]))
        else:
            mAP = 0
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, None


from timm.utils import accuracy
@torch.no_grad()
def evaluate_imnet(model, postprocessors=None, data_loader=None, base_ds=None, device=None, output_dir=None, save_json=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output, loss_dict = model(images)
        output = output['cls_label_logits']
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, None


def getEvaluator(args):
    if args.DATA.type == "imnet":
        return evaluate_imnet
    elif args.DATA.type == "coco" or args.DATA.type == "coco_hybrid":
        return evaluate_coco
    elif args.DATA.type == "coco_mtl":
        return evaluate_coco_mtl
    else:
        raise KeyError

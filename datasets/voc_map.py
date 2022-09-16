# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Query2Label (https://github.com/SlongLiu/query2labels)
# ------------------------------------------------------------------------
import numpy as np
import os
import torch

def voc_mAP(imagessetfilelist, num, return_each=False):
    if isinstance(imagessetfilelist, str):
        imagessetfilelist = [imagessetfilelist]
    lines = []
    for imagessetfile in imagessetfilelist:
        with open(imagessetfile, 'r') as f:
            lines.extend(f.readlines())

    seg = np.array([x.strip().split(' ') for x in lines]).astype(float)
    gt_label = seg[:,num:].astype(np.int32)
    num_target = np.sum(gt_label, axis=1, keepdims = True)


    sample_num = len(gt_label)
    class_num = num
    tp = np.zeros(sample_num)
    fp = np.zeros(sample_num)
    valid = np.zeros(class_num)
    aps = []

    for class_id in range(class_num):
        confidence = seg[:,class_id]
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        sorted_label = [gt_label[x][class_id] for x in sorted_ind]

        for i in range(sample_num):
            tp[i] = (sorted_label[i]>0)
            fp[i] = (sorted_label[i]<=0)
        true_num = 0
        true_num = sum(tp)
        if true_num > 0:
            valid[class_id] = 1
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(true_num)
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = voc_ap(rec, prec, true_num)
            aps += [ap]
        else:
            aps += [0.]

    np.set_printoptions(precision=6, suppress=True)
    aps = np.array(aps) * 100
    mAP = sum(np.array(aps) * valid) / sum(valid)
    if return_each:
        return mAP, aps
    return mAP


def voc_ap(rec, prec, true_num):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def topk_recall(imagessetfilelist, num, ks):
    if isinstance(imagessetfilelist, str):
        imagessetfilelist = [imagessetfilelist]
    lines = []
    for imagessetfile in imagessetfilelist:
        with open(imagessetfile, 'r') as f:
            lines.extend(f.readlines())

    seg = np.array([x.strip().split(' ') for x in lines]).astype(float)
    pred_logits = seg[:, :num]  # ds * classes
    gt_label = seg[:,num:].astype(np.int32)  # ds * classes
    num_target = np.sum(gt_label, axis=1, keepdims = True) # ds * 1

    sorted_inds = np.argsort(-pred_logits, axis=-1)
    results = {}
    pred_onehot = np.zeros_like(gt_label) # ds * classes
    for k in ks:
        indexs = sorted_inds[:, :k] # ds * k
        np.put_along_axis(pred_onehot, indices=sorted_inds[:, :k], values=1, axis=1)
        intersection = pred_onehot * gt_label
        recall = intersection.sum() / gt_label.sum()
        results[k] = recall
    return results


def thr_recall(imagessetfilelist, num, ks):
    if isinstance(imagessetfilelist, str):
        imagessetfilelist = [imagessetfilelist]
    lines = []
    for imagessetfile in imagessetfilelist:
        with open(imagessetfile, 'r') as f:
            lines.extend(f.readlines())

    seg = np.array([x.strip().split(' ') for x in lines]).astype(float)
    pred_logits = seg[:, :num]  # ds * classes
    gt_label = seg[:,num:].astype(np.int32)  # ds * classes

    results = {}
    counts = {}
    for k in ks:
        pred_onehot = pred_logits > k
        intersection = pred_onehot * gt_label
        recall = intersection.sum() / gt_label.sum()
        results[k] = recall
        counts[k] = (pred_onehot.sum(axis=1).mean(), intersection.sum(axis=1).mean(), gt_label.sum(axis=1).mean())
    save_dir = os.path.split(imagessetfilelist[0])[0]
    torch.save({
        "pred_probs": pred_logits,
        "gt_label": gt_label,
        "recall": results,
        "counts": counts,
    }, os.path.join(save_dir, "thr_recall.pth"))
    return results, counts

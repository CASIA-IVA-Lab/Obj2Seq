import json
import os, cv2,sys
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from match import matchOneCls

gt_file = '/root/coco17/annotations/person_keypoints_val2017.json'
pd_kps_file = 'checkpoints/DetrKeypoint/0503_e2e4box_min1/results_keypoints.json'
img_dir = '/root/coco17/val2017'
save_dir = 'checkpoints/DetrKeypoint/0503_e2e4box_min1/results_kps'
THR = 0.2
CAT_IDS = None#[5, 6, 17, 18, 21, 22, 28, 32, 36, 41, 47, 49, 61, 63, 76, 81, 87]
skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

if not os.path.exists(save_dir):
   os.mkdir(save_dir)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.2


def matchKeypoints(gts, pds, iou):
    len_gt, len_pd = gts.shape[0], pds.shape[0]
    matched_gt, matched_pd = np.zeros(len_gt), np.zeros(len_pd)
    for idx_pd in range(len_pd):
        idx_gts = np.argsort(iou[idx_pd])[::-1]
        for idx_gt in idx_gts:
            if matched_gt[idx_gt]:
                break
            if iou[idx_pd, idx_gt] < 0.5:
                break
            matched_pd[idx_pd]=1
            matched_gt[idx_gt]=1
            break
    return matched_gt, matched_pd


def drawImg(img_id, gt_ann, pd_kps, coco_eval):
    if CAT_IDS is None:
        gt_ids = gt_ann.getAnnIds(imgIds = [img_id])
    else:
        gt_ids = gt_ann.getAnnIds(imgIds = [img_id], catIds=CAT_IDS)
    if len(gt_ids) < 1:
        return
    gt_anns = [gt_ann.anns[i] for i in gt_ids]
    gts = np.asarray([[ann["category_id"], *ann["bbox"]] for ann in gt_anns])
    gt_cls = np.asarray([ann["category_id"] for ann in gt_anns])
    gt_box = np.asarray([ann["bbox"] for ann in gt_anns]).reshape((-1,4))
    gt_keypoints = np.asarray([ann["keypoints"] for ann in gt_anns]).reshape((-1,17, 3))
    len_gt = gt_box.shape[0]
    gt_matched = np.zeros(len_gt)
    
    if CAT_IDS is None:
        pd_ids = pd_kps.getAnnIds(imgIds = [img_id])
    else:
        pd_ids = pd_kps.getAnnIds(imgIds = [img_id], catIds=CAT_IDS)
    pd_anns = [pd_kps.anns[i] for i in pd_ids]
    pd_score = np.asarray([ann["score"] for ann in pd_anns])

    # 首先，先确定maxDet以内的
    inds = np.argsort( -pd_score, kind='mergesort')
    pd_anns = [pd_anns[i] for i in inds]

    pd_score = np.asarray([ann["score"] for ann in pd_anns])
    pd_cls = np.asarray([ann["category_id"] for ann in pd_anns])
    pd_keypoints = np.asarray([ann["keypoints"] for ann in pd_anns]).reshape((-1,17, 3))

    # 把超过阈值的留下来
    above_thr = np.where(pd_score > THR)
    pd_cls, pd_score = pd_cls[above_thr], pd_score[above_thr]
    pd_keypoints = pd_keypoints[above_thr]

    len_pd = pd_keypoints.shape[0]
    if len_pd == 0:
        return

    iou_table = coco_eval.ious[(img_id, 1)][:len_pd] # 20, ngts
    cls_matched_gt, cls_matched_pd = matchKeypoints(gt_keypoints.reshape((-1, 17, 3)), pd_keypoints.reshape((-1, 17, 3)), iou_table)
    gt_matched = cls_matched_gt
    pd_matched = cls_matched_pd

    img_file = gt_ann.imgs[img_id]["file_name"]
    img_wholepath = os.path.join(img_dir, img_file)
    image = cv2.imread(img_wholepath)
    gt_box[:, 2:] = gt_box[:, :2] + gt_box[:, 2:]

    for cls_name, box, keypoint, matched in zip(gt_cls, gt_box, gt_keypoints, gt_matched):
        color = (255, 0, 0) if matched else (255, 255, 255)
        box, cls_name = box.astype(np.int32), str(cls_name)
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color)
        ((txt_w, txt_h), _) = cv2.getTextSize(cls_name, font, font_scale, 1)
        txt_st, txt_ed = (box[0], box[3]-txt_h), (box[0]+txt_w, box[3])
        cv2.rectangle(image, txt_st, txt_ed, color, -1)
        for ipt in keypoint:
            cv2.circle(image, (ipt[0], ipt[1]), 3, color)
        cv2.putText(image, cls_name, (box[0], box[3]), font, font_scale, (0, 0, 0), lineType=cv2.LINE_AA)

    for cls_name, keypoint, matched, score in zip(pd_cls, pd_keypoints, pd_matched, pd_score):
        color = (0, 255, 0) if matched else (0, 0, 255)
        cls_name = str(cls_name)
        keypoint = keypoint.astype(np.int32)
        text = '%s %.3f'%(cls_name, score)
        ((txt_w, txt_h), _) = cv2.getTextSize(text, font, font_scale, 1)
        txt_st, txt_ed = (keypoint[0, 0], keypoint[0, 1]-txt_h), (keypoint[0, 0]+txt_w, keypoint[0, 1])
        cv2.rectangle(image, txt_st, txt_ed, color, -1)
        for isk in skeleton:
            pt1 = keypoint[isk[0] - 1, :2]
            pt2 = keypoint[isk[1] - 1, :2]
            cv2.line(image, pt1, pt2, color, 1)
        for ipt in keypoint:
            cv2.circle(image, (ipt[0], ipt[1]), 3, color)
        cv2.putText(image, text, (keypoint[0, 0], keypoint[0, 1]), font, font_scale, (0, 0, 0), lineType=cv2.LINE_AA)

    img_savepath = os.path.join(save_dir, img_file)
    cv2.imwrite(img_savepath, image)


if __name__ == "__main__":
    gt = COCO(gt_file)
    pd_kps = gt.loadRes(pd_kps_file)
    coco_eval = COCOeval(gt, pd_kps, "keypoints")
    coco_eval.evaluate()
    for img_id in gt.imgs:
        drawImg(img_id, gt, pd_kps, coco_eval)
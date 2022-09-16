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
Transforms and data augmentation for both image + bbox.
"""
import random
import numpy as np

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

from util.box_ops import box_xyxy_to_cxcywh
from util.misc import interpolate


def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    target["size"] = torch.tensor([h, w])

    fields = ["labels", "area", "iscrowd"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    if "masks" in target:
        # FIXME should we update the area here if there are no boxes?
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    if "keypoints" in target:
        keypoints = target['keypoints']
        keypoints = keypoints - torch.as_tensor([j, i, 0])
        target['keypoints'] = keypoints
        fields.append("keypoints")

    # remove elements for which the boxes or masks that have zero area
    if "boxes" in target or "masks" in target:
        # favor boxes selection when defining which elements to keep
        # this is compatible with previous implementation
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


KPT_FLIP_INDEX = [
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
]


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    if "keypoints" in target:
        keypoints = target['keypoints']
        keypoints = keypoints * torch.as_tensor([-1, 1, 1]) + torch.as_tensor([w, 0, 0])
        keypoints = keypoints[:, KPT_FLIP_INDEX]
        target['keypoints'] = keypoints

    return flipped_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if w < h:
           ow = size
           oh = int(size * h / w)
           if max_size is not None and oh > max_size:
               oh = max_size
               ow = int( max_size * w / h)
        else:
            oh = size
            ow = int(size * w / h)
            if max_size is not None and ow > max_size:
               ow = max_size
               oh = int( max_size * h / w)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    if "keypoints" in target:
        keypoints = target["keypoints"]
        scaled_keypoints = keypoints * torch.as_tensor([ratio_width, ratio_height, 1])
        target["keypoints"] = scaled_keypoints

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    if "keypoints" in target:
        keypoints = target['keypoints']
        keypoints = keypoints + torch.as_tensor([padding[0], padding[1], 0])
        target['keypoints'] = keypoints
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
    """

    def __init__(self, output_size=1333, aug_scale_min=0.3, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def rescale_target(self, scaled_size, image_size, target):
        # compute rescaled targets
        image_scale = scaled_size / image_size
        ratio_height, ratio_width = image_scale

        target = target.copy()
        target["size"] = scaled_size

        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        if "masks" in target:
            masks = target['masks']
            masks = interpolate(
                masks[:, None].float(), scaled_size, mode="nearest")[:, 0] > 0.5
            target['masks'] = masks

        if "keypoints" in target:
            keypoints = target["keypoints"]
            scaled_keypoints = keypoints * torch.as_tensor([ratio_width, ratio_height, 1])
            target["keypoints"] = scaled_keypoints
        return target

    def crop_target(self, region, target):
        i, j, h, w = region
        fields = ["labels", "area", "iscrowd"]

        target = target.copy()
        target["size"] = torch.tensor([h, w])

        if "boxes" in target:
            boxes = target["boxes"]
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
            cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)
            area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
            target["boxes"] = cropped_boxes.reshape(-1, 4)
            target["area"] = area
            fields.append("boxes")

        if "masks" in target:
            # FIXME should we update the area here if there are no boxes?
            target['masks'] = target['masks'][:, i:i + h, j:j + w]
            fields.append("masks")

        if "keypoints" in target:
            keypoints = target['keypoints']
            keypoints = keypoints - torch.as_tensor([j, i, 0])
            target['keypoints'] = keypoints
            fields.append("keypoints")

        # remove elements for which the boxes or masks that have zero area
        if "boxes" in target or "masks" in target:
            # favor boxes selection when defining which elements to keep
            # this is compatible with previous implementation
            if "boxes" in target:
                cropped_boxes = target['boxes'].reshape(-1, 2, 2)
                keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
            else:
                keep = target['masks'].flatten(1).any(1)

            for field in fields:
                target[field] = target[field][keep]
        return target

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
        if "keypoints" in target:
            keypoints = target['keypoints']
            keypoints = keypoints + torch.as_tensor([padding[1], padding[0], 0])
            target['keypoints'] = keypoints
        return target

    def __call__(self, image, target=None):
        image_size = image.size
        image_size = torch.tensor(image_size[::-1])

        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().int()

        scaled_image = F.resize(image, scaled_size.tolist())

        if target is not None:
            target = self.rescale_target(scaled_size, image_size, target)

        # randomly crop or pad images
        if random_scale > 1:
            # Selects non-zero random offset (x, y) if scaled image is larger than desired_size.
            max_offset = scaled_size - out_desired_size
            offset = (max_offset * torch.rand(2)).floor().int()
            region = (offset[0].item(), offset[1].item(),
                      out_desired_size[0].item(), out_desired_size[1].item())
            output_image = F.crop(scaled_image, *region)
            if target is not None:
                target = self.crop_target(region, target)
        else:
            padding = out_desired_size - scaled_size
            output_image = F.pad(scaled_image, [0, 0, padding[1].item(), padding[0].item()])
            if target is not None:
                target = self.pad_target(padding, target)

        return output_image, target


class RandomDistortion(object):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, prob=0.5):
        self.prob = prob
        self.tfm = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, target=None):
        if np.random.random() < self.prob:
            return self.tfm(img), target
        else:
            return img, target


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = box_xyxy_to_cxcywh(boxes)
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes
        if "keypoints" in target:
            keypoints = target["keypoints"]
            keypoints = keypoints / torch.tensor([w, h, 1], dtype=torch.float32)
            target["keypoints"] = keypoints
        return image, target


class GenerateClassificationResults(object):
    def __init__(self, num_cats):
        self.num_cats = num_cats

    def __call__(self, image, target):
        multi_labels = target["labels"].unique()
        multi_label_onehot = torch.zeros(self.num_cats)
        multi_label_onehot[multi_labels] = 1
        multi_label_weights = torch.ones_like(multi_label_onehot)

        # filter crowd items
        keep = target["iscrowd"] == 0
        fields = ["labels", "area", "iscrowd"]
        if "boxes" in target:
            fields.append("boxes")
        if "masks" in target:
            fields.append("masks")
        if "keypoints" in target:
            fields.append("keypoints")
        for field in fields:
            target[field] = target[field][keep]

        if 'neg_category_ids' in target:
            # TODO:LVIS
            not_exhaustive_category_ids = [self.json_category_id_to_contiguous_id[idx] for idx in img_info['not_exhaustive_category_ids'] if idx in self.json_category_id_to_contiguous_id]
            neg_category_ids = [self.json_category_id_to_contiguous_id[idx] for idx in img_info['neg_category_ids'] if idx in self.json_category_id_to_contiguous_id]
            multi_label_onehot[not_exhaustive_category_ids] = 1
            multi_label_weights = multi_label_onehot.clone()
            multi_label_weights[neg_category_ids] = 1
        else:
            sample_prob = torch.zeros_like(multi_label_onehot) - 1
            sample_prob[target["labels"].unique()] = 1
        target["multi_label_onehot"] = multi_label_onehot
        target["multi_label_weights"] = multi_label_weights
        target["force_sample_probs"] = sample_prob

        return image, target


class RearrangeByCls(object):
    def __init__(self, keep_keys=["size", "orig_size", "image_id", "multi_label_onehot", "multi_label_weights", "force_sample_probs"], min_keypoints_train=0):
        # TODO: min_keypoints_train is deperacated
        self.min_keypoints_train = min_keypoints_train
        self.keep_keys = keep_keys

    def __call__(self, image, target=None):
        target["class_label"] = target["labels"].unique()

        new_target = {}
        for icls in target["labels"].unique():
            icls = icls.item()
            new_target[icls] = {}
            where = target["labels"] == icls

            new_target[icls]["boxes"] = target["boxes"][where]
            if icls == 0 and "keypoints" in target:
                new_target[icls]["keypoints"] = target["keypoints"][where]

        for key in self.keep_keys:
            new_target[key] = target[key]
        return image, new_target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string

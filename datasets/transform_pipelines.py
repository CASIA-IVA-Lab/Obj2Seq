# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------

import datasets.transforms as T
import random
import numpy as np
from PIL import ImageDraw
from torchvision.transforms import transforms
from randaugment import RandAugment

def make_coco_transforms(image_set, args):
    # args.DATA.TRANSFORM
    input_size = args.input_size
    max_input_size = args.max_input_size

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    if input_size == 800:
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
    elif input_size == 480:
        scales = [256, 288, 320, 352, 384, 416, 448, 480, 512 ,544]
    else:
        raise NotImplemented

    if image_set == 'train':
        transforms = [T.RandomHorizontalFlip()]
        if args.large_scale_jitter:
            transforms.append(T.LargeScaleJitter(output_size=max_input_size))
        else:
            transforms.append(T.RandomSelect(
                T.RandomResize(scales, max_size=max_input_size),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_input_size),
                ])
            ))
        if args.color_jitter:
            transforms.append(T.RandomDistortion(0.5, 0.5, 0.5, 0.5))
        if args.arrange_by_class:
            transforms.extend([
                normalize,
                T.GenerateClassificationResults(num_cats=80),
                T.RearrangeByCls(min_keypoints_train=args.min_keypoints_train),
            ])
        else:
            transforms.append(normalize)
        return T.Compose(transforms)

    if image_set == 'val':
        transforms = [
            T.RandomResize([input_size], max_size=max_input_size),
            normalize,
        ]
        if args.arrange_by_class:
            transforms.extend([
                T.GenerateClassificationResults(num_cats=80),
                T.RearrangeByCls(),
            ])
        return T.Compose(transforms)

    raise ValueError(f'unknown {image_set}')


from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def make_imnet_transforms(image_set, args):
    # args.DATA.TRANSFORM
    is_train = image_set == 'train'
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=None,
            auto_augment=None,
            interpolation='bicubic',
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


def make_mtl_transforms(image_set, args):
    if image_set == "train":
        return transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor(),
            # normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            # normalize, # no need, toTensor does normalization
        ])


def build_transform(image_set, args):
    # _C.DATA.TRANSFORM
    if args.fix_transform:
        image_set = "val"
    if args.type == "coco":
        return make_coco_transforms(image_set, args)
    elif args.type == "coco_mtl":
        return make_mtl_transforms(image_set, args)
    elif args.type == "imnet":
        return make_coco_transforms(image_set, args)
    else:
        raise KeyError

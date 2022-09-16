# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------

import json
import os

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MixReader:
    """
    Support fs_opencv and ceph
    """

    def __init__(self, image_dir, color_mode, conf_path="~/petreloss.conf"):
        self.image_dir = image_dir
        self.color_mode = color_mode
        
        if conf_path:
            from petrel_client.client import Client

            self.cclient = Client(conf_path)
        
        assert color_mode in ["RGB", "BGR", "GRAY"], "{} not supported".format(
            color_mode
        )
        if color_mode != "BGR":
            self.cvt_color = getattr(cv2, "COLOR_BGR2{}".format(color_mode))
        else:
            self.cvt_color = None

    def ceph_read(self, filename, image_type):
        img_bytes = self.cclient.get(filename)
        assert img_bytes is not None
        img_mem_view = memoryview(img_bytes)
        img_array = np.frombuffer(img_mem_view, np.uint8)
        result = cv2.imdecode(img_array, image_type)
        return result

    def __call__(self, filename):
        if self.image_dir is not None:
            filename = os.path.join(self.image_dir, filename)
        # ceph
        if filename.startswith("s3://"):
            img = self.ceph_read(filename, cv2.IMREAD_COLOR)
        # fs_opencv
        else:
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
        if self.color_mode != "BGR":
            img = cv2.cvtColor(img, self.cvt_color)
        return img


class CustomDataset(Dataset):
    def __init__(self, meta_file, image_dir=None, color_mode="RGB", transform=None):
        self.parse_meta_file(meta_file)
        self.image_reader = MixReader(image_dir, color_mode)
        self.transform = transform

    def parse_meta_file(self, meta_file):

        with open(meta_file) as f:
            lines = f.readlines()

        self.num = len(lines)
        self.metas = []
        for line in lines:
            try:
                info = json.loads(line)
                filename = info["filename"]
                label = int(info["label"]) if "label" in info else 0
            except:
                info = line.split(" ")
                filename = info[0]
                label = int(info[1]) if len(info) >= 2 else 0
            self.metas.append((filename, label))

        print(f"  == total images: {len(self.metas)}")

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename, label = self.metas[idx]
        img = self.image_reader(filename)
        if self.transform is not None:
            img = self.transform(Image.fromarray(img))
        return img, label


import torch
from pathlib import Path
import os, cv2
import torchvision.datasets as datasets
from pycocotools.coco import COCO
from PIL import Image

class CocoDetection(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        target = target.max(dim=0)[0]
        return img, target


def build_coco_mtl(image_set, transform, args):
    # args.DATA
    root = args.COCO.coco_path
    ann_file = args.COCO.anno_train if image_set == "train" else args.COCO.anno_val
    if str(root)[:3] != "s3:":
        root = Path(root)
        assert root.exists(), f'provided COCO path {root} does not exist'
        img_root = root if (root/"val2017").exists() else (root / "images")
    else:
        img_root = root
    img_folder = os.path.join(img_root, f"{image_set}2017")

    dataset = CocoDetection(img_folder, ann_file, transform=transform)
    return dataset
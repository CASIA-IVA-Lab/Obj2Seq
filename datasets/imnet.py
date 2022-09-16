import os
from torchvision import datasets

try:
    from .reader import CustomDataset
except:
    print("fail to import Sensetime Reader")

def build_dataset(prefix, transform, args):
    root = os.path.join(args.IMNET.data_path, prefix)
    if "s3://" in args.IMNET.data_path:
        meta_file = os.path.join(args.IMNET.meta_file, '{}.txt'.format('train' if is_train else 'val'))
        return CustomDataset(meta_file=meta_file,image_dir=root,color_mode="RGB",transform=transform,)
    else:
        return datasets.ImageFolder(root, transform=transform)

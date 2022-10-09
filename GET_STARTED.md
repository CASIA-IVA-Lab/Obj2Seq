**Obj2Seq**: Formatting Objects as Sequences with Class Prompt for Visual Tasks
========

## Installation
First, create a new conda environment. We suggest you to install pytorch 1.8.
```
conda create -n obj2seq python==3.7
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```
Then, clone the repository locally and install dependencies:
```
git clone https://github.com/CASIA-IVA-Lab/Obj2Seq.git
pip install -r requirements.txt
```
Compile MultiScaleDeformableAttention from Deformable-DETR. (If you have complied it in other repository, please ignore this step.)
```
cd models/ops
bash ./make.sh
```

## Data Preparation

Link path to coco2017 to data/coco
```
mkdir data
ln -s /path/to/coco data/coco
```
or modify data path in config files
```
DATA:
  COCO:
    coco_path: /path/to/coco
    anno_train: /path/to/coco_train_json_file
    anno_val: /path/to/coco_val_json_file
```

## Prompt Generation

We provide CLIP-initialied class prompts [here](word_arrays/). If prompts for other sets of categories are required, please follow this section.

1. Prepare COCO-like json file.

2. Run the command below to generate class prompts embeddings.
```
python scripts/dump_clip_features.py --ann JSON_FILE_PATH --out_path OUTPUT_PATH
```
For example
```
python scripts/dump_clip_features.py \
       --ann data/coco/annotations/instances_val2017.json \
       --out_path word_arrays/coco_clip_v2.npy
```

## Training
To train with slurm on multiple nodes:
```
bash scripts/run_slurm.sh NUM_NODES /path/to/config /path/to/output/dir [OTHER_ARGS]
```
For example, to train Obj2Seq on 2 nodes:
```
bash scripts/run_slurm.sh 2 configs/detection_r50.yaml checkpoints/detection_r50
```

We also provide scripts for pytorch distributed training:
```
bash run.sh /path/to/config /path/to/output/dir
```

Before running, you may need to modify `DATA.batch_size` (number of images on each GPU) in config according to your GPUs.

### Evaluation
To evaluate Obj2Seq on a single node with 8 GPUs:
```
bash run.sh /path/to/config /path/to/output/dir --eval [--resume /path/to/checkpoint.pth] 
```
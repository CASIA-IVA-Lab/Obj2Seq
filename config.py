# ------------------------------------------------------------------------
# Obj2Seq: Formatting Objects as Sequences with Class Prompt for Visual Tasks
# Copyright (c) 2022 CASIA & Sensetime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Swin Transformer (https://github.com/megvii-research/AnchorDETR)
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License
# --------------------------------------------------------

import os
import yaml
import copy
from yacs.config import CfgNode as CN


# -----------------------------------------------------------------------------
# Config Settings
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
_C.DATA.type = 'coco'
_C.DATA.batch_size = 1
_C.DATA.num_workers = 2
_C.DATA.distributed = True
_C.DATA.cache_mode = False # whether to cache images on memory
_C.DATA.sampler_fix_split = False

_C.DATA.TRANSFORM = CN()
_C.DATA.TRANSFORM.type = "coco"
_C.DATA.TRANSFORM.fix_transform = False
# transforms for COCO
_C.DATA.TRANSFORM.input_size = 800
_C.DATA.TRANSFORM.max_input_size = 1333
_C.DATA.TRANSFORM.large_scale_jitter = False
_C.DATA.TRANSFORM.color_jitter = False
_C.DATA.TRANSFORM.num_classes = 80
_C.DATA.TRANSFORM.arrange_by_class = True
_C.DATA.TRANSFORM.min_keypoints_train = 0

_C.DATA.COLLECT_FN = CN()
_C.DATA.COLLECT_FN.type = 'none' # [COCObyCLS, none]
_C.DATA.COLLECT_FN.fix_input = None
_C.DATA.COLLECT_FN.input_divisor = None

_C.DATA.COCO = CN()
_C.DATA.COCO.coco_path = 'data/coco'
_C.DATA.COCO.anno_train = 'data/coco/annotations/instances_train2017.json'
_C.DATA.COCO.anno_val = 'data/coco/annotations/instances_val2017.json'
_C.DATA.COCO.remove_empty_annotations = False
_C.DATA.COCO.masks = False

_C.DATA.COCO_HYBRID = CN()
_C.DATA.COCO_HYBRID.coco_path = 'data/coco'
_C.DATA.COCO_HYBRID.detection_anno = 'data/coco/annotations/instances_{}2017.json'
_C.DATA.COCO_HYBRID.keypoint_anno = 'data/coco/annotations/person_keypoints_{}2017.json'

_C.DATA.IMNET = CN()
_C.DATA.IMNET.data_path = ''
_C.DATA.IMNET.meta_file = '/mnt/lustre/chenzhiyang.vendor/meta'


# basic layer config
BASIC_LAYER_CFG = CN()
BASIC_LAYER_CFG.hidden_dim = 256
BASIC_LAYER_CFG.nheads = 8
BASIC_LAYER_CFG.dim_feedforward = 1024
BASIC_LAYER_CFG.dropout = 0.
BASIC_LAYER_CFG.self_attn_dropout = 0.
BASIC_LAYER_CFG.activation = "relu"
BASIC_LAYER_CFG.pre_norm = False
# some removal
BASIC_LAYER_CFG.no_self_attn = False
BASIC_LAYER_CFG.cross_attn_no_value_proj = False
# for Deformable-DETR like
BASIC_LAYER_CFG.n_levels = 4
BASIC_LAYER_CFG.n_points = 4

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.hidden_dim = 256
_C.MODEL.pretrained = ""
_C.MODEL.fixed_params = []
# Model type
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.backbone = 'resnet50'
_C.MODEL.BACKBONE.train_backbone = True
_C.MODEL.BACKBONE.num_feature_levels = 4
_C.MODEL.BACKBONE.RESNET = CN()
_C.MODEL.BACKBONE.RESNET.name = 'resnet50'
_C.MODEL.BACKBONE.RESNET.dilation = False
_C.MODEL.BACKBONE.RESNET.pretrained = True
_C.MODEL.BACKBONE.RESNET.norm_layer = 'FrozenBN'
_C.MODEL.BACKBONE.SWIN = CN()
_C.MODEL.BACKBONE.SWIN.embed_dim = 96
_C.MODEL.BACKBONE.SWIN.depths = [2, 2, 6, 2]
_C.MODEL.BACKBONE.SWIN.num_heads = [3, 6, 12, 24]
_C.MODEL.BACKBONE.SWIN.window_size = 7
_C.MODEL.BACKBONE.SWIN.pretrained = None

_C.MODEL.enc_layers = 6
_C.MODEL.ENCODER_LAYER = copy.deepcopy(BASIC_LAYER_CFG)
# position embedding TODO: A more generalized pos emb
_C.MODEL.hidden_dim = 256
_C.MODEL.position_embedding = 'sine'
_C.MODEL.num_feature_levels = 4

# prompt indicator
_C.MODEL.with_prompt_indicator = True
_C.MODEL.PROMPT_INDICATOR = CN()
_C.MODEL.PROMPT_INDICATOR.num_blocks = 2
_C.MODEL.PROMPT_INDICATOR.return_intermediate = True
_C.MODEL.PROMPT_INDICATOR.level_preserve = [] # only for deformable, empty means all feature levels are used
# cfg for attention layer
_C.MODEL.PROMPT_INDICATOR.BLOCK = copy.deepcopy(BASIC_LAYER_CFG)
_C.MODEL.PROMPT_INDICATOR.BLOCK.no_self_attn = True
# cfg for prompt vectors
_C.MODEL.PROMPT_INDICATOR.CLASS_PROMPTS = CN()
_C.MODEL.PROMPT_INDICATOR.CLASS_PROMPTS.num_classes = 80
_C.MODEL.PROMPT_INDICATOR.CLASS_PROMPTS.init_vectors = "" # .npy or .pth file, empty means random initialized
_C.MODEL.PROMPT_INDICATOR.CLASS_PROMPTS.fix_class_prompts = False
# cfg for classifier
_C.MODEL.PROMPT_INDICATOR.CLASSIFIER = CN()
_C.MODEL.PROMPT_INDICATOR.CLASSIFIER.type = 'dict'
_C.MODEL.PROMPT_INDICATOR.CLASSIFIER.hidden_dim = 256
_C.MODEL.PROMPT_INDICATOR.CLASSIFIER.num_layers = 2
_C.MODEL.PROMPT_INDICATOR.CLASSIFIER.init_prob = 0.1
_C.MODEL.PROMPT_INDICATOR.CLASSIFIER.num_points = 1
_C.MODEL.PROMPT_INDICATOR.CLASSIFIER.skip_and_init = False
_C.MODEL.PROMPT_INDICATOR.CLASSIFIER.normalize_before = False
# asl loss
_C.MODEL.PROMPT_INDICATOR.LOSS = CN()
_C.MODEL.PROMPT_INDICATOR.LOSS.losses = ['asl']
_C.MODEL.PROMPT_INDICATOR.LOSS.asl_optimized = True
_C.MODEL.PROMPT_INDICATOR.LOSS.asl_loss_weight = 0.25
_C.MODEL.PROMPT_INDICATOR.LOSS.asl_gamma_pos = 0.0
_C.MODEL.PROMPT_INDICATOR.LOSS.asl_gamma_neg = 2.0
_C.MODEL.PROMPT_INDICATOR.LOSS.asl_clip = 0.0
# cfg for retention_policy
_C.MODEL.PROMPT_INDICATOR.retain_categories = True
_C.MODEL.PROMPT_INDICATOR.RETENTION_POLICY = CN()
_C.MODEL.PROMPT_INDICATOR.RETENTION_POLICY.train_max_classes = 20
_C.MODEL.PROMPT_INDICATOR.RETENTION_POLICY.train_min_classes = 20
_C.MODEL.PROMPT_INDICATOR.RETENTION_POLICY.train_class_thr = 0.0
_C.MODEL.PROMPT_INDICATOR.RETENTION_POLICY.eval_min_classes = 20
_C.MODEL.PROMPT_INDICATOR.RETENTION_POLICY.eval_max_classes = 20
_C.MODEL.PROMPT_INDICATOR.RETENTION_POLICY.eval_class_thr = 0.0


_C.MODEL.with_object_decoder = True
_C.MODEL.OBJECT_DECODER = CN()
_C.MODEL.OBJECT_DECODER.LAYER = copy.deepcopy(BASIC_LAYER_CFG)
_C.MODEL.OBJECT_DECODER.num_layers = 4
_C.MODEL.OBJECT_DECODER.num_query_position = 100
_C.MODEL.OBJECT_DECODER.spatial_prior = 'sigmoid'
_C.MODEL.OBJECT_DECODER.refine_reference_points = False
_C.MODEL.OBJECT_DECODER.with_query_pos_embed = False
# OUTPUT Layers
_C.MODEL.OBJECT_DECODER.HEAD = CN()
_C.MODEL.OBJECT_DECODER.HEAD.type = "SeparateDetectHead"
_C.MODEL.OBJECT_DECODER.HEAD.sg_previous_logits = False
_C.MODEL.OBJECT_DECODER.HEAD.combine_method = "none"
# for sequence head
_C.MODEL.OBJECT_DECODER.HEAD.pos_emb = True
_C.MODEL.OBJECT_DECODER.HEAD.num_steps = 4
_C.MODEL.OBJECT_DECODER.HEAD.num_classes = 80
_C.MODEL.OBJECT_DECODER.HEAD.task_category = "configs/tasks/coco_detection.json"
## for change structure in attention
_C.MODEL.OBJECT_DECODER.HEAD.self_attn_proj = True
_C.MODEL.OBJECT_DECODER.HEAD.cross_attn_no_value_proj = True
_C.MODEL.OBJECT_DECODER.HEAD.no_ffn = True
## to deperacate
_C.MODEL.OBJECT_DECODER.HEAD.keypoint_output = "nd_box_relative"

# if single classifier
_C.MODEL.OBJECT_DECODER.HEAD.CLASSIFIER = CN()
_C.MODEL.OBJECT_DECODER.HEAD.CLASSIFIER.type = 'dict'
_C.MODEL.OBJECT_DECODER.HEAD.CLASSIFIER.hidden_dim = 256
_C.MODEL.OBJECT_DECODER.HEAD.CLASSIFIER.num_layers = 2
_C.MODEL.OBJECT_DECODER.HEAD.CLASSIFIER.init_prob = 0.01
_C.MODEL.OBJECT_DECODER.HEAD.CLASSIFIER.num_points = 1
_C.MODEL.OBJECT_DECODER.HEAD.CLASSIFIER.skip_and_init = False
_C.MODEL.OBJECT_DECODER.HEAD.CLASSIFIER.normalize_before = False

_C.MODEL.OBJECT_DECODER.HEAD.LOSS = CN()
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.num_classes = 80
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.losses = ['labels', 'boxes']
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.aux_loss = True
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.focal_alpha = 0.25
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.cls_loss_coef = 2.0
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.bbox_loss_coef = 5.0
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.giou_loss_coef = 2.0
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.mse_loss_coef = 0.0
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.keypoint_l1_loss_coef = 1.0
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.keypoint_oks_loss_coef = 1.0
# more options for class loss
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.bce_negative_weight = 1.0
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.class_normalization = "num_box"  # ["num_box", "num_pts", "mean", "none"]

# for keypoints
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.keypoint_criterion = "L1"
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.keypoint_normalization = "num_box"  # ["num_box", "num_pts", "mean", "none"]
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.oks_normalization = "num_box"
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.keypoint_reference = "absolute" # ["absolute" or "relative"]
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.keypoint_relative_ratio = 1.0

_C.MODEL.OBJECT_DECODER.HEAD.LOSS.MATCHER = CN()
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.MATCHER.fix_match_train = ""
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.MATCHER.fix_match_val = ""
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.MATCHER.set_class_type = "focal" # ["focal", "bce", "logits", "probs"]
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.MATCHER.set_cost_class = 2.0
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.MATCHER.set_cost_bbox = 5.0
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.MATCHER.set_cost_giou = 2.0
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.MATCHER.set_cost_keypoints_oks = 0.0
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.MATCHER.set_cost_keypoints_l1 = 0.0
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.MATCHER.set_class_normalization = "none"
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.MATCHER.set_box_normalization = "none" # ["num_box", "num_pts", "mean", "none"]
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.MATCHER.set_keypoint_normalization = "none"
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.MATCHER.set_oks_normalization = "none"
#### maybe this is deprecated ?
_C.MODEL.OBJECT_DECODER.HEAD.LOSS.MATCHER.set_keypoint_reference = "absolute" # ["absolute" or "relative"]

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.epochs = 50
_C.TRAIN.clip_max_norm = 0.1 # gradient clipping max norm
_C.TRAIN.lr = 1e-4
_C.TRAIN.lr_groups = [["backbone"]]
_C.TRAIN.lr_mults = [0.1]
_C.TRAIN.weight_decay = 1e-4
_C.TRAIN.no_weight_decay_keywords = ["class_prompts", "position", "pos_emb"]
_C.TRAIN.lr_drop = 40
_C.TRAIN.lr_drop_epochs = None # not used
_C.TRAIN.sgd = False
# timm scheduler
_C.TRAIN.sched = "Step"

_C.EVAL = CN()
_C.EVAL.postprocessor = "MultiClass"
_C.EVAL.iou_types = ["bbox"]
_C.EVAL.save_json = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)
    post_process(config)
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)
    args.DATA = config.DATA
    args.MODEL = config.MODEL
    args.TRAIN = config.TRAIN
    args.EVAL = config.EVAL

    return config

def post_process(config):
    # fix dilation config
    dilation = config.MODEL.BACKBONE.RESNET.dilation
    if isinstance(dilation, str):
        if dilation.lower() == 'false':
            config.MODEL.BACKBONE.RESNET.dilation = False
        elif dilation.lower() == 'true':
            config.MODEL.BACKBONE.RESNET.dilation = True
        else:
            raise ValueError("The dilation should be True or False")
    # SeqDetectHead needs config for attention layer
    if "Seq" in config.MODEL.OBJECT_DECODER.HEAD.type:
        old_no_proj = config.MODEL.OBJECT_DECODER.HEAD.cross_attn_no_value_proj
        config.MODEL.OBJECT_DECODER.HEAD.update(config.MODEL.OBJECT_DECODER.LAYER)
        config.MODEL.OBJECT_DECODER.HEAD.cross_attn_no_value_proj = old_no_proj

        config.MODEL.OBJECT_DECODER.HEAD.LOSS.task_category = config.MODEL.OBJECT_DECODER.HEAD.task_category
        config.MODEL.OBJECT_DECODER.HEAD.LOSS.num_classes = config.MODEL.OBJECT_DECODER.HEAD.num_classes

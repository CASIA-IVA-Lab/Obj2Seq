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
import os
import copy
import argparse
import datetime
import json
import random
import time
from pathlib import Path
from functools import partial

import numpy as np
import torch
import datasets
from datasets.coco_eval import evaluate
import util.misc as utils
from datasets import build_dataset, build_dataloader, get_coco_api_from_dataset
from engine import getEvaluator, train_one_epoch
from models import build_model, build_postprocessor
from config import get_config


def get_args_parser():
    parser = argparse.ArgumentParser('Obj2Seq', add_help=False)

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # dataset parameters
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='/data/detr-workdir/r50-dc5',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', default=False, action='store_true', help='whether to resume from last checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    # add for distributed
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training') # Add for HPC
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # add for dataset
    parser.add_argument('--eval_interval', default=10, type=int)
    # config file
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    return parser


def build_optimizer(cfg_train, model_without_ddp, steps_per_epoch=None):
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    def no_weight_decay_func(name, param):
        if len(param.shape) == 1 or name.endswith(".bias"):
            return True
        out = False
        for key in cfg_train.no_weight_decay_keywords:
            if key in name:
                print("no weight decay for " + name)
                out = True
                break
        return out

    # cfg_train.lr_groups/lr_mults contains 2N length, name & multiplier
    param_dicts = []
    except_keys = []
    ## generate groups for each
    for group_keys, multiplier in zip(cfg_train.lr_groups, cfg_train.lr_mults):
        print("Params containing", group_keys, "has lr", cfg_train.lr * multiplier)
        param_dicts.append({
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if match_name_keywords(n, group_keys) and p.requires_grad and not no_weight_decay_func(n, p)],
            "lr": cfg_train.lr * multiplier,
        })
        param_dicts.append({
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if match_name_keywords(n, group_keys) and p.requires_grad and no_weight_decay_func(n, p)],
            "lr": cfg_train.lr * multiplier, "weight_decay": 0.,
        })
        except_keys += group_keys
    ## generate groups for others
    param_dicts.append({
        "params":
            [p for n, p in model_without_ddp.named_parameters()
                if not match_name_keywords(n, except_keys) and p.requires_grad and no_weight_decay_func(n, p)],
        "lr": cfg_train.lr, "weight_decay": 0.,
    })
    param_dicts.append({
        "params":
            [p for n, p in model_without_ddp.named_parameters()
                if not match_name_keywords(n, except_keys) and p.requires_grad and not no_weight_decay_func(n, p)],
        "lr": cfg_train.lr,
    })
    while len(param_dicts[0]["params"]) == 0:
        param_dicts = param_dicts[1:]

    print("Param Distribution", [len(igroup["params"]) for igroup in param_dicts])
    if cfg_train.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=cfg_train.lr, momentum=0.9,
                                    weight_decay=cfg_train.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg_train.lr,
                                      weight_decay=cfg_train.weight_decay)
    if cfg_train.sched == "OneCycle":
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg_train.lr,
                                                           steps_per_epoch=steps_per_epoch,
                                                           epochs=cfg_train.epochs, pct_start=0.2)
    elif cfg_train.sched == "Step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg_train.lr_drop)
    else:
        raise KeyError
    return optimizer, lr_scheduler


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = build_model(args)
    model.to(device)
    postprocessors = build_postprocessor(args)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    data_loader_train, data_loader_val, sampler_train = build_dataloader(dataset_train, dataset_val, args.DATA)
    evaluate = getEvaluator(args)

    optimizer, lr_scheduler = build_optimizer(args.TRAIN, model_without_ddp, steps_per_epoch=len(data_loader_train))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.DATA.type == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    elif args.DATA.type == "coco" or args.DATA.type == "coco_hybrid":
        base_ds = get_coco_api_from_dataset(dataset_val), dataset_val.prepare.contiguous_category_id_to_json_id
    else: # imnet
        base_ds = None

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    if args.auto_resume:
        if not args.resume:
            args.resume = os.path.join(args.output_dir, 'checkpoint.pth')
        if not os.path.isfile(args.resume):
            args.resume=''

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            print("Resuming from Epoch", checkpoint['epoch'])
        if args.MODEL.PROMPT_INDICATOR.CLASS_PROMPTS.fix_class_prompts:
            print("delete class_vector in resume model")
            del checkpoint['model']['transformer.prompt_indicator.class_prompts']

        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']

            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.TRAIN.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        if not args.eval:
            test_stats, coco_evaluator = evaluate(
                model, postprocessors, data_loader_val, base_ds, device, args.output_dir, args=args.EVAL
            )
    
    if args.eval:
        test_stats, coco_evaluator = evaluate(model, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, args=args.EVAL)
        if args.DATA.type == "coco" and args.output_dir:
            for iou_type in coco_evaluator.iou_types:
                utils.save_on_master(coco_evaluator.coco_eval[iou_type].eval, output_dir / f"eval_{iou_type}.pth")
            if utils.is_main_process() and args.EVAL.save_json:
                for iou_type in coco_evaluator.iou_types:
                    with open(os.path.join(args.output_dir, f'results_{iou_type}.json'), 'w') as f:
                        json.dump(coco_evaluator.results[iou_type], f)
        return

    print("Start training")
    print(args.output_dir)
    start_time = time.time()
    training_epochs = args.TRAIN.epochs
    for epoch in range(args.start_epoch, training_epochs):
        if args.distributed:
            if isinstance(sampler_train, list):
                for s in sampler_train:
                    s.set_epoch(epoch)
            else:
                sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train, optimizer,
            device, epoch, args.TRAIN.clip_max_norm, preprocessor=None)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            if (epoch + 1) % args.TRAIN.lr_drop == 0 or (epoch + 1) % 10 == 0 or (epoch + 1) % args.eval_interval == 0 or epoch >= training_epochs-5:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        if (epoch + 1) % args.eval_interval == 0 or epoch >= training_epochs-5:
            test_stats, coco_evaluator = evaluate(
                model, postprocessors, data_loader_val, base_ds, device, args.output_dir, args=args.EVAL
            )
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        
        print(args.output_dir)
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('AnchorDETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    config = get_config(args)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

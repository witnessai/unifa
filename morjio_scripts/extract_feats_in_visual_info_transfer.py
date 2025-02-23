from __future__ import division

from mmdet.datasets import DATASETS, build_dataloader
import numpy as np
from mmcv.runner.checkpoint import load_checkpoint
import argparse
import os

import torch
from mmcv import Config

from mmdet import __version__
from mmfewshot.utils import get_root_logger
from mmfewshot.detection.datasets import build_dataset
from mmfewshot.detection.models import build_detector
import ipdb 
import time 
from tqdm import tqdm

# from mmdet.apis import (get_root_logger, init_dist, set_random_seed,
#                         train_detector)
# from mmdet.datasets import build_dataset
# from mmdet.models import build_detector

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--classes', default='seen' ,help='seen or unseen classes')
    parser.add_argument('--load_from', help='the checkpoint file to load from')
    parser.add_argument('--save_dir', help='the dir to save feats and labels')
    parser.add_argument('--data_split', default='train', help='the dataset train, val, test to load from cfg file')
    parser.add_argument('--fg_iou_thr', default=0.6, help='fg iou thr > to be extracted only ')
    parser.add_argument('--bg_iou_thr', default=0.3, help='bg iou thr < to be extracted only')
    parser.add_argument('--save_file_prefix', default='', help='save file prefix')
    parser.add_argument('--save_split', default='train', help='save split')
    args = parser.parse_args()
    return args

def extract_feats(model, datasets, cfg, save_dir, data_split='train', logger=None, save_file_prefix='', save_split=''):
    
    load_checkpoint(model, cfg.load_from, 'cpu', False, logger)
    fg_th = cfg.model.train_cfg.rcnn.assigner.pos_iou_thr # 0.6
    bg_th = cfg.model.train_cfg.rcnn.assigner.neg_iou_thr # 0.3

    logger.info('load checkpoint from %s', cfg.load_from)
    logger.info(f'fg_iou_thr {fg_th} bg_iou_thr {bg_th} data_split {data_split} save_dir {save_dir}')

    model.eval()
    model = model.cuda()
    # cfg.data.imgs_per_gpu = 1
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            1,
            dist=False, 
            shuffle=False) for ds in datasets
    ]

    feats = []
    labels = []
    for index, data in enumerate(data_loaders[0]):
        bbox_feats, bbox_labels, bboxes = model.feats_extract(data['img'].data[0], data['img_metas'].data[0], data['gt_bboxes'].data[0], data['gt_labels'].data[0])
        logger.info(f"{index:05}/{len(data_loaders[0])} feats shape - {bbox_feats.shape}")

        feats.append(bbox_feats.data.cpu().numpy())
        labels.append(bbox_labels.data.cpu().numpy())
        del data, bbox_feats, bbox_labels, bboxes
    
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)

    split = f'{fg_th}_{bg_th}'

    np.save(f'{save_dir}/{save_file_prefix}{save_split}_{split}_feats.npy', feats)
    np.save(f'{save_dir}/{save_file_prefix}{save_split}_{split}_labels.npy', labels)
    # import pdb; pdb.set_trace()
    print(f"{labels.shape} num of features")

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.save_dir is not None:
        try:
            os.makedirs(args.save_dir)
        except OSError:
            pass
    cfg.work_dir = args.save_dir
    # import pdb; pdb.set_trace()
    if args.load_from is not None:
        cfg.resume_from = args.load_from
        cfg.load_from = args.load_from
    logger = get_root_logger(cfg.log_level)

    model = build_detector(
        cfg.model, logger=logger)
    
    cfg.data.train.pipeline = cfg.train_pipeline
    # cfg.data.train.classes_to_load = args.classes
    if 'test' in args.data_split:
        cfg.data.test.pipeline = cfg.train_pipeline
        cfg.test_pipeline = cfg.train_pipeline
        # cfg.data.test.classes_to_load = args.classes
        datasets = [build_dataset(cfg.data.test)]
    else:
        datasets = [build_dataset(cfg.data.train)]

    cfg.model.train_cfg.rcnn.assigner.pos_iou_thr = args.fg_iou_thr
    cfg.model.train_cfg.rcnn.assigner.min_pos_iou = args.fg_iou_thr
    cfg.model.train_cfg.rcnn.assigner.neg_iou_thr = args.bg_iou_thr
    cfg.data.imgs_per_gpu = 4

    # if 'val' in args.data_split:
    #     cfg.data.val.pipeline = cfg.train_pipeline
    #     cfg.data.val.classes_to_load = args.classes
    #     datasets = [build_dataset(cfg.data.val)]

    


    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    save_file_prefix = args.save_file_prefix
    save_split = args.save_split
    extract_feats(model, datasets, cfg, args.save_dir, data_split=args.data_split, logger=logger, save_file_prefix=save_file_prefix, save_split=save_split)


if __name__ == '__main__':
    main()

# python tools/zero_shot_utils.py configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py seen work_dirs/faster_rcnn_r101_fpn_1x_voc0712/epoch_4.pth ../../data train

# python tools/zero_shot_utils.py configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py --load_from work_dirs/faster_rcnn_r101_fpn_1x_voc0712/epoch_4.pth
# python tools/zero_shot_utils.py configs/faster_rcnn_r101_fpn_1x.py --extract_feats_only --load_from work_dirs/faster_rcnn_r101_fpn_1x/epoch_12.pth
# python tools/zero_shot_utils.py configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py --classes seen --load_from work_dirs/faster_rcnn_r101_fpn_1x_voc0712/epoch_4.pth --save_dir ../../data/voc_2 --data_split train
from __future__ import division
from socket import INADDR_ALLHOSTS_GROUP

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


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--classes', default='seen' ,help='seen or unseen classes')
    parser.add_argument('--load_from', help='the checkpoint file to load from')
    parser.add_argument('--save_dir', help='the dir to save feats and labels')
    parser.add_argument('--data_split', default='train', help='the dataset train, val, test to load from cfg file')
    parser.add_argument('--fg_iou_thr', default=0.6, help='fg iou thr > to be extracted only ')
    parser.add_argument('--bg_iou_thr', default=0.3, help='bg iou thr < to be extracted only')

    args = parser.parse_args()
    return args

def extract_feats(model, datasets, cfg, save_dir, data_split='train', logger=None):
    
    load_checkpoint(model, cfg.load_from, 'cpu', False, logger)
    fg_th = cfg.model.train_cfg.rcnn.assigner.pos_iou_thr # 0.6
    bg_th = cfg.model.train_cfg.rcnn.assigner.neg_iou_thr # 0.3

    logger.info('load checkpoint from %s', cfg.load_from)
    logger.info(f'fg_iou_thr {fg_th} bg_iou_thr {bg_th} data_split {data_split} save_dir {save_dir}')

    model.eval()
    model = model.cuda()
    # datasets is a list: [<mmdet.datasets.coco.CocoDataset object at 0x7efdd4350630>]
    cfg.data.imgs_per_gpu = 1
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            1,
            dist=False, 
            shuffle=False) for ds in datasets
    ]
    # ipdb.set_trace()

    feats = []
    labels = []
    ious = []
    soft_labels = []
    sampling_gt_labels = []
    no_sampling_gt_labels = []
    total_num = len(data_loaders[0])
    for index, data in enumerate(tqdm(data_loaders[0])):
        # bbox_feats.shape is [72, 1024]
        # bbox_labels is array([ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  7,  7,  7, 7, 16, 16, 16, 16, 16,  7,  7,  7,  7,  7,  7,  7,  7, 16, 16, 16, 16, 16, 16, 16])
        # print(data['gt_labels'].data[0])
        # time.sleep(0.25)
        bbox_feats, bbox_labels, bboxes, bbox_ious, bbox_soft_labels, bbox_sampling_gt_labels, bbox_no_sampling_gt_labels,  = model.multi_label_feats_extract(data['img'].data[0], data['img_metas'].data[0], data['gt_bboxes'].data[0], data['gt_labels'].data[0])
        # logger.info(f"{index:05}/{len(data_loaders[0])} feats shape - {bbox_feats.shape}")
        feats.append(bbox_feats.data.cpu().numpy())
        labels.append(bbox_labels.data.cpu().numpy())
        ious.append(bbox_ious.data.cpu().numpy())
        soft_labels.append(bbox_soft_labels.data.cpu().numpy())
        sampling_gt_labels.append(bbox_sampling_gt_labels.data.cpu().numpy())
        if index % 2 == 0: # to avoid value error
            no_sampling_gt_labels.append(bbox_no_sampling_gt_labels.data.cpu().numpy())
        else:
            no_sampling_gt_labels.append(bbox_no_sampling_gt_labels.data.cpu().numpy().T)
        del data, bbox_feats, bbox_labels, bboxes, bbox_soft_labels, bbox_no_sampling_gt_labels, bbox_sampling_gt_labels
        # if index == total_num//2:   
        #     feats = np.concatenate(feats)
        #     labels = np.concatenate(labels)
        #     ious = np.concatenate(ious)
        #     split = f'{fg_th}_{bg_th}'

        #     np.save(f'{save_dir}/{data_split}_{split}_feats1.npy', feats)
        #     np.save(f'{save_dir}/{data_split}_{split}_labels1.npy', labels)
        #     np.save(f'{save_dir}/{data_split}_{split}_ious1.npy', ious)
        #     np.save(f'{save_dir}/{data_split}_{split}_softlabels1.npy', soft_labels)
        #     np.save(f'{save_dir}/{data_split}_{split}_sampgtlabels1.npy', sampling_gt_labels)
        #     np.save(f'{save_dir}/{data_split}_{split}_nosampgtlabels1.npy', no_sampling_gt_labels)

        #     feats = []
        #     labels = []
        #     ious = []
        #     soft_labels = []
        #     sampling_gt_labels = []
        #     no_sampling_gt_labels = []
        # for x in multi_gt_labels: print(x.shape)
            
    feats = np.concatenate(feats)
    labels = np.concatenate(labels)
    ious = np.concatenate(ious)
    split = f'{fg_th}_{bg_th}'

    np.save(f'{save_dir}/{data_split}_{split}_feats.npy', feats)
    np.save(f'{save_dir}/{data_split}_{split}_labels.npy', labels)
    np.save(f'{save_dir}/{data_split}_{split}_ious.npy', ious)
    np.save(f'{save_dir}/{data_split}_{split}_softlabels.npy', soft_labels)
    np.save(f'{save_dir}/{data_split}_{split}_sampgtlabels.npy', sampling_gt_labels)
    np.save(f'{save_dir}/{data_split}_{split}_nosampgtlabels.npy', no_sampling_gt_labels)
    # import pdb; pdb.set_trace()
    # print(f"{labels.shape} num of features")

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
    cfg.work_dir = args.save_dir # 'data/coco/SUZSD'
    # import pdb; pdb.set_trace()
    if args.load_from is not None:
        cfg.resume_from = args.load_from # ./work_dirs/coco2014_we_use_suzsd_code_train/epoch_12.pth
        cfg.load_from = args.load_from
    logger = get_root_logger(cfg.log_level) # cfg.log_level is 'INFO'

    # cfg.model is {'type': 'FasterRCNN', 'pretrained': '../checkpoints/resnet101.pyth', 'backbone': {'type': 'ResNet', 'depth': 101, 'num_stages': 4, 'out_indices': (0, 1, 2, 3), 'frozen_stages': 1, 'style': 'pytorch'}, 'neck': {'type': 'FPN', 'in_channels': [256, 512, 1024, 2048], 'out_channels': 256, 'num_outs': 5}, 'rpn_head': {'type': 'RPNHead', 'in_channels': 256, 'feat_channels': 256, 'anchor_scales': [8], 'anchor_ratios': [0.5, 1.0, 2.0], 'anchor_strides': [4, 8, 16, 32, 64], 'target_means': [0.0, 0.0, 0.0, 0.0], 'target_stds': [1.0, 1.0, 1.0, 1.0], 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': True, 'loss_weight': 1.0}, 'loss_bbox': {'type': 'SmoothL1Loss', 'beta': 0.1111111111111111, 'loss_weight': 1.0}}, 'bbox_roi_extractor': {'type': 'SingleRoIExtractor', 'roi_layer': {'type': 'RoIAlign', 'out_size': 7, 'sample_num': 2}, 'out_channels': 256, 'featmap_strides': [4, 8, 16, 32]}, 'bbox_head': {'type': 'SharedFCBBoxHead', 'num_fcs': 2, 'in_channels': 256, 'fc_out_channels': 1024, 'roi_feat_size': 7, 'num_classes': 81, 'target_means': [0.0, 0.0, 0.0, 0.0], 'target_stds': [0.1, 0.1, 0.2, 0.2], 'reg_class_agnostic': False, 'loss_cls': {'type': 'CrossEntropyLoss', 'use_sigmoid': False, 'loss_weight': 1.0}, 'loss_bbox': {'type': 'SmoothL1Loss', 'beta': 1.0, 'loss_weight': 1.0}}}
    # cfg.train_cfg is {'rpn': {'assigner': {'type': 'MaxIoUAssigner', 'pos_iou_thr': 0.7, 'neg_iou_thr': 0.3, 'min_pos_iou': 0.3, 'ignore_iof_thr': -1}, 'sampler': {'type': 'RandomSampler', 'num': 256, 'pos_fraction': 0.5, 'neg_pos_ub': -1, 'add_gt_as_proposals': False}, 'allowed_border': 0, 'pos_weight': -1, 'debug': False}, 'rpn_proposal': {'nms_across_levels': False, 'nms_pre': 2000, 'nms_post': 2000, 'max_num': 2000, 'nms_thr': 0.7, 'min_bbox_size': 0}, 'rcnn': {'assigner': {'type': 'MaxIoUAssigner', 'pos_iou_thr': 0.5, 'neg_iou_thr': 0.5, 'min_pos_iou': 0.5, 'ignore_iof_thr': -1}, 'sampler': {'type': 'RandomSampler', 'num': 512, 'pos_fraction': 0.25, 'neg_pos_ub': -1, 'add_gt_as_proposals': True}, 'pos_weight': -1, 'debug': False}}
    # cfg.test_cfg is {'rpn': {'nms_across_levels': False, 'nms_pre': 1000, 'nms_post': 1000, 'max_num': 1000, 'nms_thr': 0.7, 'min_bbox_sizm     asd s e': 0}, 'rcnn': {'split': '65_15', 'zsd': False, 'gzsd': False, 'dataset_name': 'coco', 'score_thr': 0.05, 'nms': {'type': 'nms', 'iou_thr': 0.5}, 'max_per_img': 100}}
    model = build_detector(cfg.model, logger=logger)
    

    cfg.data.train.pipeline = cfg.train_pipeline
    datasets = [build_dataset(cfg.data.train)]

    cfg.model.train_cfg.rcnn.assigner.pos_iou_thr = args.fg_iou_thr
    cfg.model.train_cfg.rcnn.assigner.min_pos_iou = args.fg_iou_thr
    cfg.model.train_cfg.rcnn.assigner.neg_iou_thr = args.bg_iou_thr
    # cfg.data.imgs_per_gpu = 8
    cfg.data.imgs_per_gpu = 4
    if 'val' in args.data_split:
        cfg.data.val.pipeline = cfg.train_pipeline
        cfg.data.val.classes_to_load = args.classes
        datasets = [build_dataset(cfg.data.val)]

    elif 'test' in args.data_split:
    
        cfg.data.test.pipeline = cfg.train_pipeline
        datasets = [build_dataset(cfg.data.test)]
    elif 'zero_shot' in args.data_split: # args.data_split is 'zero-shot'
        cfg.data.test.pipeline = cfg.train_pipeline
        cfg.data.test.classes_to_load = args.classes # 'zero-shot'
        datasets = [build_dataset(cfg.data.test)]
    
    # cfg.checkpoint_config is {'interval': 1}
    # datasets[0].CLASSES is 80 classname 
    if cfg.checkpoint_config is not None: 
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__,
            config=cfg.text,
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES 
    # args.save_dir is data/coco/SUZSD
    # args.data_split is zero_shot
    extract_feats(model, datasets, cfg, args.save_dir, data_split=args.data_split, logger=logger)

def test():
    root_dirs = ['/home/niehui/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection/base_det/', '/home/niehui/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection/base_few_shot_det/']
    # root_dir = '/home/niehui/morjio/projects/detection/zero-shot-detection/Synthesizing_the_Unseen_for_Zero-shot_Object_Detection/mmdetection/data/coco/SUZSD'
    filenames = ['test_0.6_0.3_labels_without_remap.npy', 'train_0.6_0.3_labels_without_remap.npy']
    filenames = ['test_0.6_0.3_labels.npy', 'train_0.6_0.3_labels.npy']
    
    for root_dir in root_dirs:
        print(root_dir)
        for fn in filenames:
        # feats = np.load(os.path.join(root_dir, filenames[0]))
            labels = np.load(os.path.join(root_dir, fn))
            xx = list(set(labels))
            xx.sort()
            print(xx)
            print(len(xx))
    ipdb.set_trace()
        

if __name__ == '__main__':
    # test()
    # ipdb.set_trace()
    main()

# python tools/zero_shot_utils.py configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py seen work_dirs/faster_rcnn_r101_fpn_1x_voc0712/epoch_4.pth ../../data train

# python tools/zero_shot_utils.py configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py --load_from work_dirs/faster_rcnn_r101_fpn_1x_voc0712/epoch_4.pth
# python tools/zero_shot_utils.py configs/faster_rcnn_r101_fpn_1x.py --extract_feats_only --load_from work_dirs/faster_rcnn_r101_fpn_1x/epoch_12.pth
# python tools/zero_shot_utils.py configs/pascal_voc/faster_rcnn_r101_fpn_1x_voc0712.py --classes seen --load_from work_dirs/faster_rcnn_r101_fpn_1x_voc0712/epoch_4.pth --save_dir ../../data/voc_2 --data_split train
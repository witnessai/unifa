# Copyright (c) OpenMMLab. All rights reserved.
"""Reshape the classification and regression layer for novel classes.

The bbox head from base training only supports `num_base_classes` prediction,
while in few shot fine-tuning it need to handle (`num_base_classes` +
`num_novel_classes`) classes. Thus, the layer related to number of classes
need to be reshaped.

The original implementation provides three ways to reshape the bbox head:

    - `combine`: combine two bbox heads from different models, for example,
        one model is trained with base classes data and another one is
        trained with novel classes data only.
    - `remove`: remove the final layer of the base model and the weights of
        the removed layer can't load from the base model checkpoint and
        will use random initialized weights for few shot fine-tuning.
    - `random_init`: create a random initialized layer (`num_base_classes` +
        `num_novel_classes`) and copy the weights of base classes from the
        base model.

Temporally, we only use this script in FSCE and TFA with `random_init`.
This part of code is modified from
https://github.com/ucbdrive/few-shot-object-detection/.

Example:
    # VOC base model
    python3 -m tools.detection.misc.initialize_bbox_head \
        --src1 work_dirs/tfa_r101_fpn_voc-split1_base-training/latest.pth \
        --method random_init \
        --save-dir work_dirs/tfa_r101_fpn_voc-split1_base-training
    # COCO base model
    python3 -m tools.detection.misc.initialize_bbox_head \
        --src1 work_dirs/tfa_r101_fpn_coco_base-training/latest.pth \
        --method random_init \
        --coco \
        --save-dir work_dirs/tfa_r101_fpn_coco_base-training
"""
import argparse
import os

import torch
from mmcv.runner.utils import set_random_seed

# asd
import ipdb 

# COCO config
COCO_BASE_CLASSES = [1, 2, 3, 4, 8, 9, 15, 16, 19, 20, 24, 25, 27, 31, 35, 38, 42, 44, 50, 51, 52, 53, 55, 56, 57, 59, 60, 62, 65, 72, 73, 75, 78, 79, 82, 84, 85, 86, 90, 7, 23, 33, 34, 48, 54, 70, 74, 80]
COCO_NOVEL_CLASSES = [6, 18, 21, 22, 28, 32, 41, 47, 49, 61, 63, 76, 81, 87, 5, 17, 36]
COCO_ALL_CLASSES = sorted(COCO_BASE_CLASSES + COCO_NOVEL_CLASSES)

COCO_IDMAP = {v: i for i, v in enumerate(COCO_ALL_CLASSES)}
COCO_TAR_SIZE = 65
# LVIS config
LVIS_NOVEL_CLASSES = [
    12,  13,  16,  19,  20,  29,  30,  37,  38,  39,  41, 48,  50,  51,  62,  68,  70,  77,  81,  84,  92, 104, 105, 112, 116, 118, 122, 125, 129, 130, 135, 139, 141, 143, 146, 150, 154, 158, 160, 163, 166, 171, 178, 181, 195, 201, 208, 209, 213, 214, 221, 222, 230, 232, 233, 235, 236, 237, 239, 243, 244, 246, 249, 250, 256, 257, 261, 264, 265, 268, 269, 274, 280, 281, 286, 290, 291, 293, 294, 299, 300, 301, 303, 306, 309, 312, 315, 316, 320, 322, 325, 330, 332, 347, 348, 351, 352, 353, 354, 356, 361, 363, 364, 365, 367, 373, 375, 380, 381, 387, 388, 396, 397, 399, 404, 406, 409, 412, 413, 415, 419, 425, 426, 427, 430, 431, 434, 438, 445, 448, 455, 457, 466, 477, 478, 479, 480, 481, 485, 487, 490, 491, 502, 505, 507, 508, 512, 515, 517, 526, 531, 534, 537, 540, 541, 542, 544, 550, 556, 559, 560, 566, 567, 570, 571, 573, 574, 576, 579, 581, 582, 584, 593, 596, 598, 601, 602, 605, 609, 615, 617, 618, 619, 624, 631, 633, 634, 637, 639, 645, 647, 650, 656, 661, 662, 663, 664, 670, 671, 673, 677, 685, 687, 689, 690, 692, 701, 709, 711, 713, 721, 726, 728, 729, 732, 742, 751, 753, 754, 757, 758, 763, 768, 771, 777, 778, 782, 783, 784, 786, 787, 791, 795, 802, 804, 807, 808, 809, 811, 814, 819, 821, 822, 823, 828, 830, 848, 849, 850, 851, 852, 854, 855, 857, 858, 861, 863, 868, 872, 882, 885, 886, 889, 890, 891, 893, 901, 904, 907, 912, 913, 916, 917, 919, 924, 930, 936, 937, 938, 940, 941, 943, 944, 951, 955, 957, 968, 971, 973, 974, 982, 984, 986, 989, 990, 991, 993, 997, 1002, 1004, 1009, 1011, 1014, 1015, 1027, 1028, 1029, 1030, 1031, 1046, 1047, 1048, 1052, 1053, 1056, 1057, 1074, 1079, 1083, 1115, 1117, 1118, 1123, 1125, 1128, 1134, 1143, 1144, 1145, 1147, 1149, 1156, 1157, 1158, 1164, 1166, 1192
]
LVIS_BASE_CLASSES = [c for c in range(1203) if c not in LVIS_NOVEL_CLASSES]
LVIS_ALL_CLASSES = sorted(LVIS_BASE_CLASSES + LVIS_NOVEL_CLASSES)
LVIS_IDMAP = {v: i for i, v in enumerate(LVIS_ALL_CLASSES)}
LVIS_TAR_SIZE = 1203
# VOC config
VOC_TAR_SIZE = 20


def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument('--src1', type=str, help='Path to the main checkpoint')
    parser.add_argument(
        '--src2',
        type=str,
        default=None,
        help='Path to the secondary checkpoint. Only used when combining '
        'fc layers of two checkpoints')
    parser.add_argument(
        '--save-dir', type=str, default=None, help='Save directory')
    parser.add_argument(
        '--method',
        choices=['combine', 'remove', 'random_init'],
        required=True,
        help='Reshape method. combine = combine bbox heads from different '
        'checkpoints. remove = for fine-tuning on novel dataset, remove the '
        'final layer of the base detector. random_init = randomly initialize '
        'novel weights.')
    parser.add_argument(
        '--param-name',
        type=str,
        nargs='+',
        default=['roi_head.bbox_head.fc_cls', 'roi_head.bbox_head.fc_reg'],
        help='Target parameter names')
    parser.add_argument(
        '--tar-name',
        type=str,
        default='base_model',
        help='Name of the new checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    # Dataset
    parser.add_argument('--coco', action='store_true', help='For COCO models')
    parser.add_argument('--lvis', action='store_true', help='For LVIS models')
    return parser.parse_args()


def random_init_checkpoint(param_name, is_weight, tar_size, checkpoint, args):
    """Either remove the final layer weights for fine-tuning on novel dataset
    or append randomly initialized weights for the novel classes.

    Note: The base detector for LVIS contains weights for all classes, but only
    the weights corresponding to base classes are updated during base training
    (this design choice has no particular reason). Thus, the random
    initialization step is not really necessary.
    """
    weight_name = param_name + ('.weight' if is_weight else '.bias')
    # pretrained_weight.shape is [61, 1024]
    pretrained_weight = checkpoint['state_dict'][weight_name]
    prev_cls = pretrained_weight.size(0) # 60
    if 'fc_cls' in param_name:
        prev_cls -= 1
    if is_weight:
        feat_size = pretrained_weight.size(1)
        new_weight = torch.rand((tar_size, feat_size))
        torch.nn.init.normal_(new_weight, 0, 0.01)
    else:
        new_weight = torch.zeros(tar_size)
    if args.coco or args.lvis:
        
        BASE_CLASSES = COCO_BASE_CLASSES if args.coco else LVIS_BASE_CLASSES
        IDMAP = COCO_IDMAP if args.coco else LVIS_IDMAP
        for i, c in enumerate(BASE_CLASSES):
            idx = i if (args.coco or args.lvis) else c
            if 'fc_cls' in param_name:
                new_weight[IDMAP[c]] = pretrained_weight[idx]
            else:
                new_weight[IDMAP[c] * 4:(IDMAP[c] + 1) * 4] = \
                    pretrained_weight[idx * 4:(idx + 1) * 4]
    else:
        new_weight[:prev_cls] = pretrained_weight[:prev_cls]
    if 'fc_cls' in param_name:
        new_weight[-1] = pretrained_weight[-1]  # bg class
    checkpoint['state_dict'][weight_name] = new_weight


def combine_checkpoints(param_name, is_weight, tar_size, checkpoint,
                        checkpoint2, args):
    """Combine base detector with novel detector.

    Feature extractor weights are from the base detector. Only the final layer
    weights are combined.
    """
    if not is_weight and param_name + '.bias' not in checkpoint['state_dict']:
        return
    if not is_weight and param_name + '.bias' not in checkpoint2['state_dict']:
        return
    weight_name = param_name + ('.weight' if is_weight else '.bias')
    pretrained_weight = checkpoint['state_dict'][weight_name]
    prev_cls = pretrained_weight.size(0)
    if 'fc_cls' in param_name:
        prev_cls -= 1
    if is_weight:
        feat_size = pretrained_weight.size(1)
        new_weight = torch.rand((tar_size, feat_size))
    else:
        new_weight = torch.zeros(tar_size)
    if args.coco or args.lvis:
        BASE_CLASSES = COCO_BASE_CLASSES if args.coco else LVIS_BASE_CLASSES
        IDMAP = COCO_IDMAP if args.coco else LVIS_IDMAP
        for i, c in enumerate(BASE_CLASSES):
            idx = i if args.coco else c
            if 'fc_cls' in param_name:
                new_weight[IDMAP[c]] = pretrained_weight[idx]
            else:
                new_weight[IDMAP[c] * 4:(IDMAP[c] + 1) * 4] = \
                    pretrained_weight[idx * 4:(idx + 1) * 4]
    else:
        new_weight[:prev_cls] = pretrained_weight[:prev_cls]

    checkpoint2_weight = checkpoint2['state_dict'][weight_name]
    if args.coco or args.lvis:
        NOVEL_CLASSES = COCO_NOVEL_CLASSES if args.coco else LVIS_NOVEL_CLASSES
        IDMAP = COCO_IDMAP if args.coco else LVIS_IDMAP
        for i, c in enumerate(NOVEL_CLASSES):
            if 'fc_cls' in param_name:
                new_weight[IDMAP[c]] = checkpoint2_weight[i]
            else:
                new_weight[IDMAP[c] * 4:(IDMAP[c] + 1) * 4] = \
                    checkpoint2_weight[i * 4:(i + 1) * 4]
        if 'fc_cls' in param_name:
            new_weight[-1] = pretrained_weight[-1]
    else:
        if 'fc_cls' in param_name:
            new_weight[prev_cls:-1] = checkpoint2_weight[:-1]
            new_weight[-1] = pretrained_weight[-1]
        else:
            new_weight[prev_cls:] = checkpoint2_weight
    checkpoint['state_dict'][weight_name] = new_weight
    return checkpoint


def reset_checkpoint(checkpoint):
    if 'scheduler' in checkpoint:
        del checkpoint['scheduler']
    if 'optimizer' in checkpoint:
        del checkpoint['optimizer']
    if 'iteration' in checkpoint:
        checkpoint['iteration'] = 0


def main():
    args = parse_args()
    set_random_seed(args.seed)
    checkpoint = torch.load(args.src1)
    save_name = args.tar_name + f'_{args.method}_bbox_head.pth'
    save_dir = args.save_dir \
        if args.save_dir != '' else os.path.dirname(args.src1)
    save_path = os.path.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    reset_checkpoint(checkpoint)

    if args.coco:
        TAR_SIZE = COCO_TAR_SIZE
    elif args.lvis:
        TAR_SIZE = LVIS_TAR_SIZE
    else:
        TAR_SIZE = VOC_TAR_SIZE

    if args.method == 'remove':
        # Remove parameters
        for param_name in args.param_name:
            del checkpoint['state_dict'][param_name + '.weight']
            if param_name + '.bias' in checkpoint['state_dict']:
                del checkpoint['state_dict'][param_name + '.bias']
    elif args.method == 'combine':
        checkpoint2 = torch.load(args.src2)
        tar_sizes = [TAR_SIZE + 1, TAR_SIZE * 4]
        for idx, (param_name,
                  tar_size) in enumerate(zip(args.param_name, tar_sizes)):
            combine_checkpoints(param_name, True, tar_size, checkpoint,
                                checkpoint2, args)
            combine_checkpoints(param_name, False, tar_size, checkpoint,
                                checkpoint2, args)
    elif args.method == 'random_init':
        tar_sizes = [TAR_SIZE + 1, TAR_SIZE * 4]
        for idx, (param_name,
                  tar_size) in enumerate(zip(args.param_name, tar_sizes)):
            random_init_checkpoint(param_name, True, tar_size, checkpoint,
                                   args)
            random_init_checkpoint(param_name, False, tar_size, checkpoint,
                                   args)
    else:
        raise ValueError(f'not support method: {args.method}')

    torch.save(checkpoint, save_path)
    print('save changed checkpoint to {}'.format(save_path))


if __name__ == '__main__':
    main()

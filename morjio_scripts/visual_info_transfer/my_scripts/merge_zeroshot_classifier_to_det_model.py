import os 
import ipdb 
import numpy as np
import torch
from mmcv.runner.utils import set_random_seed
import argparse


COCO_TAR_SIZE = 80

# novel classes cat id 通过COCO_IDMAP转换后的label id是[ 4, 15, 28, 29, 48, 61, 64]
COCO_ZERO_SHOT_CLASSES = [5, 17, 33, 34, 54, 70, 74]
# wrong cat id:[7, 14, 17, 23, 36, 48, 70, 80]
#
# 
COCO_BASE_CLASSES = [1, 2, 3, 4, 6, 8, 9, 10, 11, 13, 15, 16, 18, 19, 20, 21, 22, 24, 25, 27, 28, 31, 32, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 67, 72, 73, 75, 76, 77, 78, 79, 81, 82, 84, 85, 86, 87, 88, 90]
# all classes cat id 
COCO_ALL_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
COCO_IDMAP = {cat_id: idx for idx, cat_id in enumerate(COCO_ALL_CLASSES)}

# most similar to zero shot in base and few shot
ZERO_MAP_BASE_FEW = {4:2, 15:16, 28:24, 29:16, 48:53, 61:79, 64:66}
# wrong map: 6:5, 12:67, 15:16, 21:20, 31:30, 42:44, 61:79, 70:68


Objects365_TAR_SIZE = 365
Objects365_ZERO_SHOT_CLASSES = ()
Objects365_BASE_CLASSES = ()
Objects365_IDMAP = dict()


def parse_args():
    parser = argparse.ArgumentParser()
    # Paths
    parser.add_argument(
        '--src1', 
        type=str, 
        # default='work_dirs/asd_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth',
        default='work_dirs/rahman_ce_asd_65_8_7_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth',
        help='Path to the main checkpoint')
    parser.add_argument(
        '--src2',
        type=str,
        default='checkpoints/asd_65_8_7/zeroshot_classifier_upper_bound/classifier_best.pth',
        # default='checkpoints/asd_65_8_7/zeroshot_classifier_upper_bound/classifier_best.pth',
        # default='checkpoints/asd_65_8_7/classifier_best.pth',
        # default='checkpoints/asd_65_8_7/classifier_best_from_suzsd.pth',
        help='Path to the secondary checkpoint. Only used when combining '
        'fc layers of two checkpoints')
    parser.add_argument(
        '--save-dir', 
        type=str, 
        default='checkpoints/asd_65_8_7/merged_det_model/', 
        help='Save directory')
    parser.add_argument(
        '--param-name',
        type=str,
        nargs='+',
        default=['roi_head.bbox_head.fc_cls', 'roi_head.bbox_head.fc_reg'],
        help='Target parameter names')
    parser.add_argument(
        '--tar-name',
        type=str,
        default='merged_base_few_zero_shot_det_model_upper_bound.pth',
        help='Name of the new checkpoint')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    # Dataset
    parser.add_argument('--coco', action='store_true', help='For COCO models')
    parser.add_argument('--objects365', action='store_true', help='For Objects365 models')
    return parser.parse_args()

def combine_checkpoints(param_name, is_weight, tar_size, base_few_shot_det, zero_shot_classifier, args):
    if not is_weight and param_name + '.bias' not in base_few_shot_det['state_dict']:
        return
    if not is_weight and  'fc1.bias' not in zero_shot_classifier['state_dict']:
        return
    weight_name = param_name + ('.weight' if is_weight else '.bias')
    base_few_shot_det_weight = base_few_shot_det['state_dict'][weight_name]
    weight_name = 'fc1' + ('.weight' if is_weight else '.bias')
    zero_shot_classifier_weight = zero_shot_classifier['state_dict'][weight_name]
    NOVEL_CLASSES = COCO_ZERO_SHOT_CLASSES if args.coco else Objects365_ZERO_SHOT_CLASSES   
    # BASE_CLASSES = COCO_BASE_CLASSES if args.coco else Objects365_BASE_CLASSES   
    IDMAP = COCO_IDMAP if args.coco else Objects365_IDMAP 
    for i, c in enumerate(NOVEL_CLASSES): #忽略背景类
        if 'fc_cls' in param_name:
            # print(base_few_shot_det_weight[[4, 15, 28, 29, 48, 61, 64], :])
            base_few_shot_det_weight[IDMAP[c]] = zero_shot_classifier_weight[i]
            # print(base_few_shot_det_weight[[4, 15, 28, 29, 48, 61, 64], :])
            # ipdb.set_trace()
        else: 
            # 零样本类回归分支的参数全部赋予person类回归分支的参数
            # 改进方案：赋予基类+小样本类中语义最相似类别的回归分支参数
            base_few_shot_det_weight[IDMAP[c] * 4:(IDMAP[c] + 1) * 4] = \
                base_few_shot_det_weight[ZERO_MAP_BASE_FEW[IDMAP[c]] * 4:(ZERO_MAP_BASE_FEW[IDMAP[c]] + 1) * 4]
    
def main():
    args = parse_args()
    set_random_seed(args.seed)
    save_path = os.path.join(args.save_dir, args.tar_name)
    assert (args.coco != False) or (args.objects365 != False) 
    if args.coco:
        TAR_SIZE = COCO_TAR_SIZE
    elif args.objects365:
        TAR_SIZE = Objects365_TAR_SIZE
    tar_sizes = [TAR_SIZE+1, TAR_SIZE*4]
    # 检测模型选择基类+小样本类的预训练检测器
    base_few_shot_det = torch.load(args.src1)

    ## save old parameters for checking
    old_cls_weight = base_few_shot_det['state_dict']['roi_head.bbox_head.fc_cls.weight'].numpy().copy()
    old_cls_bias = base_few_shot_det['state_dict']['roi_head.bbox_head.fc_cls.bias'].numpy().copy()
    old_reg_weight = base_few_shot_det['state_dict']['roi_head.bbox_head.fc_reg.weight'].numpy().copy()
    old_reg_bias = base_few_shot_det['state_dict']['roi_head.bbox_head.fc_reg.bias'].numpy().copy()


    zero_shot_classifier = torch.load(args.src2)
    for idx, (param_name, tar_size) in enumerate(zip(args.param_name, tar_sizes)):
        # print(base_few_shot_det['state_dict']['roi_head.bbox_head.fc_cls.weight'][[4, 15, 28, 29, 48, 61, 64], :])
        combine_checkpoints(param_name, True, tar_size, base_few_shot_det,
                                zero_shot_classifier, args)
        combine_checkpoints(param_name, False, tar_size, base_few_shot_det,
                                zero_shot_classifier, args)


    ## save new parameters for checking
    new_cls_weight = base_few_shot_det['state_dict']['roi_head.bbox_head.fc_cls.weight'].numpy()
    new_cls_bias = base_few_shot_det['state_dict']['roi_head.bbox_head.fc_cls.bias'].numpy()
    new_reg_weight = base_few_shot_det['state_dict']['roi_head.bbox_head.fc_reg.weight'].numpy()
    new_reg_bias = base_few_shot_det['state_dict']['roi_head.bbox_head.fc_reg.bias'].numpy()
    ## checking
    print('cls_weight============================================')
    print(old_cls_weight[[4, 15, 28, 29, 48, 61, 64], :])
    print(new_cls_weight[[4, 15, 28, 29, 48, 61, 64], :])
    print('cls_bias============================================')
    print(old_cls_bias[[4, 15, 28, 29, 48, 61, 64]])
    print(new_cls_bias[[4, 15, 28, 29, 48, 61, 64]])
    print('reg_weight============================================')
    old_reg_weight = old_reg_weight.reshape(-1, 4, 1024)
    new_reg_weight = new_reg_weight.reshape(-1, 4, 1024)
    print(old_reg_weight[[4, 15, 28, 29, 48, 61, 64], :, :])
    print(new_reg_weight[[4, 15, 28, 29, 48, 61, 64], :, :])
    print('reg_bias============================================')
    old_reg_bias = old_reg_bias.reshape(-1, 4)
    new_reg_bias = new_reg_bias.reshape(-1, 4)
    print(old_reg_bias[[4, 15, 28, 29, 48, 61, 64], :])
    print(new_reg_bias[[4, 15, 28, 29, 48, 61, 64], :])

    merged_base_few_zero_shot_det = base_few_shot_det
    torch.save(merged_base_few_zero_shot_det, save_path)
    print('save changed checkpoint to {}'.format(save_path))
    

if __name__ == '__main__':
    # for i, c in enumerate(COCO_ZERO_SHOT_CLASSES):
    #     print(COCO_IDMAP[c])
    # ipdb.set_trace()
    main()
# 运行脚本的命令
#  python morjio_scripts/train_feature_generator/my_codes/merge_zeroshot_classifier_to_det_model.py  --coco

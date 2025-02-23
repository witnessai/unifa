import torch
import torch.nn as nn
import ipdb 
import os 
import numpy as np
from tqdm import tqdm

np.set_printoptions(suppress=True)
# torch.set_printoptions(precision=3,sci_mode=False)
## load pretrained det model 
model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/work_dirs/fsd_65_15_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth'
# model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/work_dirs/rahman_ce_asd_65_8_7_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth'
checkpoint = torch.load(model_path)
COCO_NOVEL_CLASSES_AND_BG = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78, 80] # len(COCO_NOVEL_CLASSES_AND_BG) is 16
COCO_IDMAP = {v: i for i, v in enumerate(COCO_NOVEL_CLASSES_AND_BG)}
param_name = 'roi_head.bbox_head.fc_cls'
is_weight_list = [True, False]
tar_size = [16, 1024]
new_weight = nn.Linear(tar_size[1], tar_size[0])
for is_weight in is_weight_list:
    weight_name = param_name + ('.weight' if is_weight else '.bias')
    pretrained_weight = checkpoint['state_dict'][weight_name] # shape is [81, 1024] or [i81]
    for idx, c in enumerate(COCO_NOVEL_CLASSES_AND_BG):
        if is_weight:
            new_weight.weight[COCO_IDMAP[c]] = pretrained_weight[c]
        else:
            new_weight.bias[COCO_IDMAP[c]] = pretrained_weight[c]
model_det = new_weight
model_det = model_det.cuda()
model_det.eval()

## load pretrained cls model
model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_65_15/0.6725_best_acc_in_testdata_20230208_text_embedding/classifier_best_latest.pth'
is_weight_list = [True, False]
tar_size = [16, 1024]
new_weight = nn.Linear(tar_size[1], tar_size[0])
checkpoint = torch.load(model_path)
# ipdb.set_trace()
param_name = 'fc1'
for is_weight in is_weight_list:
    weight_name = param_name + ('.weight' if is_weight else '.bias')
    pretrained_weight = checkpoint['state_dict'][weight_name] # shape is [81, 1024] or [81]
    for i in range(tar_size[0]):
        if is_weight:
            new_weight.weight[i] = pretrained_weight[i]
        else:
            new_weight.bias[i] = pretrained_weight[i]
model_cls = new_weight
model_cls = model_cls.cuda()
model_cls.eval()

## use few-shot classifier to classify test feats

data_root = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split'
feat_path = os.path.join(data_root, 'test_0.6_0.3_feats.npy')
features = np.load(feat_path) # # type(feats) is np, feats.shape is (288849, 1024)
label_path = os.path.join(data_root, 'test_0.6_0.3_labels.npy')
labels = np.load(label_path) # type(labels) is np, labels.shape is (288849)
features = torch.from_numpy(features)
features = features.cuda()
labels = torch.from_numpy(labels)
labels = labels.cuda()
labels[labels==80] = 15
# id_map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 80:15}

train_num = labels.shape[0]
inst_total = np.zeros(tar_size[0])
right_total = np.zeros(tar_size[0])
wrong_total = np.zeros(tar_size[0])
for i in tqdm(range(train_num)):
    input = features[i]
    output_det = torch.softmax(model_det(input), 0)
    output_cls = torch.softmax(model_cls(input), 0)
    val_det = torch.max(output_det)
    val_cls = torch.max(output_cls)
    gt = labels[i]
    idx_det = torch.argmax(output_det)
    idx_cls = torch.argmax(output_cls)
    if val_det > val_cls:
        idx = idx_det
    else:
        idx = idx_cls
    # idx = idx_det
    # ipdb.set_trace()
    if gt == idx:
        inst_total[gt] += 1
        right_total[gt] += 1
    else:
        inst_total[gt] += 1
        wrong_total[idx] += 1

acc_total = np.zeros(tar_size[0])
for i in range(len(inst_total)):
    acc_total[i] = 1.0*right_total[i]/inst_total[i]
print(acc_total)
print(right_total)
print(inst_total)
ipdb.set_trace()




# shell command:
# CUDA_VISIBLE_DEVICES=5 python merge_det_and_cls_classifier_inference_results_on_testset.py






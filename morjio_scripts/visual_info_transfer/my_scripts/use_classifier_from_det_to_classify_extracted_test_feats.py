import torch
import torch.nn as nn
import ipdb 
import os 
import numpy as np
from tqdm import tqdm

np.set_printoptions(suppress=True)
## load pretrained det model to extract few-shot classifier
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

## use few-shot classifier to classify test feats
model = new_weight
data_root = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split'
feat_path = os.path.join(data_root, 'test_0.6_0.3_feats.npy')
features = np.load(feat_path) # # type(feats) is np, feats.shape is (288849, 1024)
label_path = os.path.join(data_root, 'test_0.6_0.3_labels.npy')
labels = np.load(label_path) # type(labels) is np, labels.shape is (288849)

model = model.cuda()
features = torch.from_numpy(features)
features = features.cuda()
labels = torch.from_numpy(labels)
labels = labels.cuda()

# id_map = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 80:15}
labels[labels==80] = 15


train_num = labels.shape[0]
inst_total = np.zeros(tar_size[0])
right_total = np.zeros(tar_size[0])
wrong_total = np.zeros(tar_size[0])
for i in tqdm(range(train_num)):
    input = features[i]
    output = model(input)
    gt = labels[i]
    idx = torch.argmax(output)
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

# output:
# acc_total:
# [0.00084664 0.00079713 0.00391134 0.00107964 0.00144578 0.00017615
# 0.01575859 0.         0.00724152 0.         0.00018818 0.00271346
#  0.16472507 0.0125261  0.05896806 0.00098148]
# pred_total:
# [47101.  8323.  8774.  4071. 62073.  5948. 35573.  5209.  4762.  2809.
#  2614.  2678. 34888. 21440. 41262.   174.]
# inst_total:
# [  8268.  10036.   2301.  12041.   4150.  11354.   5965.   4202.   7457.
#   8961.   5314.  11056.   4292.    479.    407. 192566.]



# shell command:
# CUDA_VISIBLE_DEVICES=5 python use_classifier_from_det_to_classify_extracted_test_feats.py






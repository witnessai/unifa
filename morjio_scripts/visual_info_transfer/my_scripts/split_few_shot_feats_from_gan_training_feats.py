import os
import ipdb 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
             discriminant_analysis, random_projection)


train_feats = np.load('/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/train_0.6_0.3_feats.npy')
train_labels = np.load('/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/train_0.6_0.3_labels.npy')

# 背景类的数量过多，只需要提取一部分即可，但也尽量远大于前景类别数量
COCO_NOVEL_CLASSES = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78]
COCO_NOVEL_CLASSES_AND_BG = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78, 80] 

# print(np.sum(train_labels==80)) # 背景数量 output is 5759142
# print(len(train_labels)) # 所有实例数量 8638713

few_shot_and_bg_feats = []
few_shot_and_bg_labels = []

for idx in COCO_NOVEL_CLASSES_AND_BG:
    few_shot_and_bg_feats.append(train_feats[train_labels==idx])
    few_shot_and_bg_labels.append(train_labels[train_labels==idx])
    # if idx is not 80: # 如果不是背景类，则提取全部特征和标签
    #     few_shot_and_bg_feats.append(train_feats[train_labels==idx])
    #     few_shot_and_bg_labels.append(train_labels[train_labels==idx])
    # else: # 如果是背景类，则保留所有标签？
    #     few_shot_and_bg_feats.append(train_feats[train_labels==idx][:10000])
    #     few_shot_and_bg_labels.append(train_labels[train_labels==idx][:10000])
feats = np.concatenate(few_shot_and_bg_feats)
labels = np.concatenate(few_shot_and_bg_labels)

fg_th = 0.6
bg_th = 0.3
split = f'{fg_th}_{bg_th}'
data_split = 'few_shot_15_and_bg_train'
save_dir = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split'
np.save(f'{save_dir}/{data_split}_{split}_feats.npy', feats)
np.save(f'{save_dir}/{data_split}_{split}_labels.npy', labels)
# import pdb; pdb.set_trace()
print(f"{labels.shape} num of features") # (5760044,) num of features

ipdb.set_trace()
import enum
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import FeaturesCls
from torch.utils.data import DataLoader
from splits import get_asd_base_class_ids 
import ipdb 
from tqdm import tqdm


class ClsModelTrain(nn.Module):
    def __init__(self, num_classes=4):
        super(ClsModelTrain, self).__init__()
        self.fc1 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
    def forward(self, feats=None, classifier_only=False):
        x = self.fc1(feats)
        return x

# 初始化分类器
classifier = ClsModelTrain(num_classes=66) # 65+1
classifier = classifier.cuda()
# optimizer = optim.Adam(classifier.parameters(), lr=0.0001, betas=(0.5, 0.999))

# 赋值基类检测器的分类器参数
dataset = 'coco'
classes_split = '65_8_7'
classes_to_train = np.concatenate((get_asd_base_class_ids(dataset, split=classes_split), [80]))   
pretrained_det_path = 'work_dirs/asd_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth'
pretrained_det = torch.load(pretrained_det_path)
cls_weight = pretrained_det['state_dict']['roi_head.bbox_head.fc_cls.weight']
cls_bias = pretrained_det['state_dict']['roi_head.bbox_head.fc_cls.bias']
for i, cls_id in enumerate(classes_to_train):
    classifier.fc1.weight[i] = cls_weight[cls_id]
    classifier.fc1.bias[i] = cls_bias[cls_id]

# 加载数据集
batch_size = 32
root = 'data/coco/any_shot_detection/base_det'
base_split = 'train_0.6_0.3'
features = np.load(f"{root}/{base_split}_feats.npy")
labels = np.load(f"{root}/{base_split}_labels.npy")
dataset = FeaturesCls(features=features, labels=labels, split='base', classes_to_train=classes_to_train) 
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)

classifier.eval() # 只做测试，看看对提取的基类特征分类的准确率如何
correct_num = 0
total_num = 0
for iter, (in_feat, in_label)  in enumerate(tqdm(dataloader)):
    in_feat = in_feat.type(torch.float).cuda()
    in_label = in_label.cuda()
    preds = classifier(feats=in_feat, classifier_only=True)
    value, ind = torch.max(preds, 1)
    total_num += preds.shape[0]
    correct_num += torch.sum(in_label == ind)
print(correct_num*1.0/total_num)
ipdb.set_trace()
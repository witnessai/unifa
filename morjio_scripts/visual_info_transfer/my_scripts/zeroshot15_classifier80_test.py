import enum
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import FeaturesCls
from torch.utils.data import DataLoader
from splits import get_asd_zero_shot_class_ids, get_asd_few_zero_shot_class_ids, get_asd_base_few_zero_shot_class_ids
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
num_classes = 16
classifier = ClsModelTrain(num_classes=num_classes)
classifier = classifier.cuda()
# optimizer = optim.Adam(classifier.parameters(), lr=0.0001, betas=(0.5, 0.999))

# 赋值基类检测器的分类器参数
dataset = 'coco'
classes_split = '65_15'
if num_classes == 81:
    classes_to_train = np.concatenate((get_asd_base_few_zero_shot_class_ids(dataset, split=classes_split), [80]))
elif num_classes == 16:
    classes_to_train = np.concatenate((get_asd_few_zero_shot_class_ids(dataset, split=classes_split), [80]))
  
# pretrained_cls_path = 'checkpoints/fsd_65_80/0.028_best_acc_in_testdata_20230419_text_embedding/classifier_latest.pth'
pretrained_cls_path = 'checkpoints/fsd_65_80/2023-04-19-16:11:09/classifier_best_latest.pth'

pretrained_cls = torch.load(pretrained_cls_path)
cls_weight = pretrained_cls['state_dict']['fc1.weight']
cls_bias = pretrained_cls['state_dict']['fc1.bias']

for i, cls_id in enumerate(classes_to_train):
    classifier.fc1.weight[i] = cls_weight[cls_id]
    classifier.fc1.bias[i] = cls_bias[cls_id]
# 加载数据集
batch_size = 32
root = 'data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/tfa/10shot'
# root = 'data/coco/any_shot_detection/base_few_shot_det/multi_labels/zero_shot/'
base_split = 'test_0.6_0.3'
features = np.load(f"{root}/{base_split}_feats.npy")
labels = np.load(f"{root}/{base_split}_labels.npy")

print(len(labels))
# unique_label = list(set(labels))
# for lab in unique_label:
#     print(lab, '  ', len(labels[labels==lab]))
# ipdb.set_trace()
# ipdb.set_trace()
if num_classes == 16:
    classes_to_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 80])
elif num_classes == 81:
    before_map_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 80]
    before_map_list.sort(reverse=True)
    after_map_list = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78, 80]
    after_map_list.sort(reverse=True)
    for idx, c in enumerate(before_map_list):
        labels[labels == c] = after_map_list[idx]
    classes_to_train = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80])
dataset = FeaturesCls(features=features, labels=labels, split='base', classes_to_train=classes_to_train) 
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)

classifier.eval() 
correct_num = 0
total_num = 0
values_list = []
pred_all = []
gt_all = []

# check logit of base
few_15 = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78]
base_65 = [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 79, 80]
base_65_mean_max_logit = 0
few_15_mean_max_logit = 0 
base_65_max_logit = 0
base_65_max_logit_list = []

for iter, (in_feat, in_label)  in enumerate(tqdm(dataloader)):
    in_feat = in_feat.type(torch.float).cuda()
    in_label = in_label.cuda()
    preds = classifier(feats=in_feat, classifier_only=True)

    if num_classes == 81:
        base_65_mean_max_logit = (torch.max(preds[:, base_65], dim=1)[0].mean().item() + iter * base_65_mean_max_logit) / (iter + 1)
        few_15_mean_max_logit = (torch.max(preds[:, few_15], dim=1)[0].mean().item() + iter * few_15_mean_max_logit) / (iter + 1)
        base_65_max_logit = max(torch.max(preds[:, base_65]), base_65_max_logit)
        base_65_max_logit_list.append(torch.max(preds[:, base_65], dim=1)[0])
    preds = torch.softmax(preds, 1)
    value, ind = torch.max(preds, 1)
    values_list.extend(list(value.cpu().detach().numpy()))
    total_num += preds.shape[0]
    correct_num += torch.sum(in_label == ind)
    pred_all.append(ind.cpu().numpy())
    gt_all.append(in_label.cpu().numpy())
pred_all = np.concatenate(pred_all)
gt_all = np.concatenate(gt_all)
if num_classes == 16:
    class_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]) 
    acc_per_class = np.zeros(16)
elif num_classes == 81:
    class_labels = np.array([4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78, 80]) 
    acc_per_class = np.zeros(16)

for index, label in enumerate(class_labels):
    idx = (gt_all == label)
    acc_per_class[index] = np.sum(pred_all[idx] == gt_all[idx]) / np.sum(idx)
    print(f'Class {label} has {np.sum(idx)} samples, acc is {acc_per_class[index]}')
print("mean acc: ", np.mean(acc_per_class))
print('acc: ', (correct_num*1.0/total_num).item()*100, '%')
print(sum(values_list)/(len(values_list)))
print(few_15_mean_max_logit, base_65_mean_max_logit, base_65_max_logit)

## compute std_var of max logit of base, mean is 2.0503, std_var is 2.2223
if len(base_65_max_logit_list) > 0:
    base_65_max_logit_all = torch.cat(base_65_max_logit_list)
    base_65_max_logit_stdvar = torch.sqrt(torch.sum((base_65_max_logit_all-base_65_mean_max_logit)**2)/len(base_65_max_logit_all))
    print(base_65_max_logit_stdvar)

ipdb.set_trace()

# CUDA_VISIBLE_DEVICES=1 python morjio_scripts/visual_info_transfer/my_scripts/zeroshot15_classifier80_test.py
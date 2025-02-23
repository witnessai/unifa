import enum
import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataset import FeaturesCls
from torch.utils.data import DataLoader
from splits import get_asd_base_class_ids, get_asd_zero_shot_class_ids
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
classifier = ClsModelTrain(num_classes=8) # 65+1
classifier = classifier.cuda()
# optimizer = optim.Adam(classifier.parameters(), lr=0.0001, betas=(0.5, 0.999))

# 赋值基类检测器的分类器参数
dataset = 'coco'
classes_split = '65_8_7'
classes_to_train = np.concatenate((get_asd_zero_shot_class_ids(dataset, split=classes_split), [80]))  
# pretrained_cls_path = 'checkpoints/asd_65_8_7/classifier_best_from_suzsd_v3_fix_order_bg_bias.pth'
# pretrained_cls_path = 'checkpoints/asd_65_8_7/all80_classifier_upper_bound/classifier_best_reorder_zs_cls.pth' 
pretrained_cls_path = 'checkpoints/asd_65_8_7/unseen15_classifier_upper_bound/train_2nd/classifier_best_reorder_zs_cls.pth'
# pretrained_cls_path = 'checkpoints/asd_65_8_7/zeroshot_classifier_upper_bound/train_2nd/classifier_best.pth' 
# pretrained_cls_path = 'checkpoints/asd_65_8_7/zeroshot_classifier_upper_bound/classifier_best.pth' 
# pretrained_cls_path = 'checkpoints/asd_65_8_7/classifier_best.pth'
# pretrained_cls_path = 'checkpoints/asd_65_8_7/2022-05-09-few-zero-shot/classifier_best_reorder_zs_cls.pth'
# pretrained_cls_path = 'checkpoints/asd_65_8_7/2022-05-11-zero-shot/classifier_best.pth'
# pretrained_cls_path = 'checkpoints/asd_65_8_7/2022-05-16-zero-shot-fixrandomseed/classifier_best.pth'
# pretrained_cls_path = 'checkpoints/asd_65_8_7/2022-05-11-base-few-zero-shot/classifier_best_reorder_zs_cls.pth'
# ipdb.set_trace()
pretrained_cls = torch.load(pretrained_cls_path)
cls_weight = pretrained_cls['state_dict']['fc1.weight']
cls_bias = pretrained_cls['state_dict']['fc1.bias']
for i, cls_id in enumerate(classes_to_train):
    classifier.fc1.weight[i] = cls_weight[i]
    classifier.fc1.bias[i] = cls_bias[i]
if cls_bias.shape[0] == 16: # 如果是src分类器是零小样本类分类器，那么需要手动将背景类参数赋值
    classifier.fc1.weight[-1] = cls_weight[-1]
    classifier.fc1.bias[-1] = cls_bias[-1]

# 加载数据集
batch_size = 32
root = 'data/coco/any_shot_detection/base_few_shot_det/'
# root = 'data/coco/any_shot_detection/base_few_shot_det/multi_labels/zero_shot/'
base_split = 'test_0.6_0.3'
features = np.load(f"{root}/{base_split}_feats.npy")
labels = np.load(f"{root}/{base_split}_labels.npy")

print(len(labels))
# unique_label = list(set(labels))
# for lab in unique_label:
#     print(lab, '  ', len(labels[labels==lab]))
# ipdb.set_trace()




dataset = FeaturesCls(features=features, labels=labels, split='base', classes_to_train=classes_to_train) 
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=4, shuffle=True, pin_memory=True)

classifier.eval() # 只做测试，看看对提取的基类特征分类的准确率如何
correct_num = 0
total_num = 0
values_list = []
for iter, (in_feat, in_label)  in enumerate(tqdm(dataloader)):
    in_feat = in_feat.type(torch.float).cuda()
    in_label = in_label.cuda()
    preds = classifier(feats=in_feat, classifier_only=True)
    preds = torch.softmax(preds, 1)
    value, ind = torch.max(preds, 1)
    values_list.extend(list(value.cpu().detach().numpy()))
    total_num += preds.shape[0]
    correct_num += torch.sum(in_label == ind)
print('acc: ', (correct_num*1.0/total_num).item()*100, '%')
print(sum(values_list)/(len(values_list)))
ipdb.set_trace()
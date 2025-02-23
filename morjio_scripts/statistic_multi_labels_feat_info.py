import os 
import numpy as np
import ipdb 
from tqdm import tqdm

root_dir = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection/base_few_shot_det/multi_labels/zero_shot/'

filename = 'train_0.6_0.3_labels_without_remap.npy'
path = os.path.join(root_dir, filename)
maxioulabels = np.load(path, allow_pickle=True)

filename = 'train_0.6_0.3_softlabels.npy'
path = os.path.join(root_dir, filename)
softlabels = np.load(path, allow_pickle=True)

filename = 'train_0.6_0.3_sampgtlabels.npy'
path = os.path.join(root_dir, filename)
sampgtlabels = np.load(path, allow_pickle=True)

filename = 'train_0.6_0.3_feats.npy'
path = os.path.join(root_dir, filename)
feats = np.load(path, allow_pickle=True)

filename = 'train_0.6_0.3_ious.npy'
path = os.path.join(root_dir, filename)
ious = np.load(path, allow_pickle=True)


print(softlabels.shape)

total_pos_cnt = 0
totol_multi_labels_cnt = np.array([0, 0, 0, 0, 0]) # 类别数量：1, 2, 3, 4, >4
for softlabel_per_img, sampgtlabel_per_img in tqdm(zip(softlabels, sampgtlabels)): # 遍历所有图片
    for softlabel_per_roi, sampgtlabel_per_roi in zip(softlabel_per_img, sampgtlabel_per_img): # 遍历所有框
        val = np.max(softlabel_per_roi)
        loc = np.argmax(softlabel_per_roi)
        unique_labels = []
        if val > 0.6:
            total_pos_cnt += 1
            unique_labels.append(sampgtlabel_per_roi[loc])
        else:
            continue
        for i, softlabel_per_ele in enumerate(softlabel_per_roi):
            if softlabel_per_ele > 0.1 and sampgtlabel_per_roi[i] not in unique_labels:
                unique_labels.append(sampgtlabel_per_roi[i])
        length = len(unique_labels)
        if length <= 4:
            totol_multi_labels_cnt[length-1] += 1
        else:
            totol_multi_labels_cnt[-1] += 1

print(sum(totol_multi_labels_cnt[1:])*1.0/total_pos_cnt)
ipdb.set_trace()
# 软标签大于0的条件：
    # 总正样本数：3423704
    # 总正样本数中多语义的个数：[1401367, 1298802,  493928,  157722,   71885] #分别表示类别数为1, 2, 3, 4, >4的情况
# 软标签大于0.1的条件：



        

from re import M
import numpy as np
import ipdb 
import os
from tqdm import tqdm

# statstic of soft labels
root_dir = 'data/coco/any_shot_detection/base_few_shot_det/multi_labels/merged/'
filename = 'test_0.6_0.3_zero_shot_labels.npy'
path = os.path.join(root_dir, filename)
data = np.load(path, allow_pickle=True)
print(set(data))
ipdb.set_trace()
np.save(path, data)

filename = 'train_0.6_0.3_feats.npy'
path = os.path.join(root_dir, filename)
data = np.load(path, allow_pickle=True)
data = np.concatenate(data)
np.save(path, data)
ipdb.set_trace()

filename = 'train_0.6_0.3_softlabels.npy'
path = os.path.join(root_dir, filename)
data = np.load(path, allow_pickle=True)
filename = 'train_0.6_0.3_labels.npy'
path = os.path.join(root_dir, filename)
pred_labels_data = np.load(path, allow_pickle=True)
filename = 'train_0.6_0.3_multigtlabels.npy'
path = os.path.join(root_dir, filename)
gt_labels_data = np.load(path, allow_pickle=True)

ipdb.set_trace()
total_cnt = 0
multi_instance_cnt = 0
for item in tqdm(data):
    for row in item:
        max = -1
        cnt = 0
        for element in row:
            if element > max:
                max = element
            if element > 0:
                cnt += 1
        if max > 0.6:
            total_cnt += 1
            if cnt >= 2:
                multi_instance_cnt += 1
print(multi_instance_cnt)
print(total_cnt)
print(multi_instance_cnt*1.0/total_cnt)
# 正样本是iou大于0.6
# soft label 大于等于2
# 测试集
# 27203
# 96252
# 0.2826226987491169
# 训练集
# 2712249
# 3429924
# 0.7907606699157183

ipdb.set_trace()

root_dir = 'data/coco/any_shot_detection/base_few_shot_det/multi_labels'
mid_dir = 'merged'
_filenames = os.listdir(root_dir)
filenames = []
for fn in _filenames:
    if 'train' in fn:
        filenames.append(fn)
iter_num = len(filenames)//2
# ipdb.set_trace()
for idx in range(iter_num):
    print(idx)
    a = filenames[idx]
    b = filenames[idx+iter_num]
    print(a, b)
    path_a = os.path.join(root_dir, a)
    path_b = os.path.join(root_dir, b)
    data_a = np.load(path_a, allow_pickle=True)
    data_b = np.load(path_b, allow_pickle=True)
    # if 'ious' in a:
    #     ipdb.set_trace()
    # data_a = np.concatenate(data_a)
    # data_b = np.concatenate(data_b)
    # data = np.concatenate([data_a, data_b])
    data = np.hstack((data_a, data_b))
    save_name = a.split('1')[0]+'.npy'
    save_path = os.path.join(root_dir, mid_dir, save_name)
    np.save(save_path, data)
    




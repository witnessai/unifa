import numpy as np
import ipdb 

root = 'data/coco/any_shot_detection/base_few_shot_det'
base_split = 'train_0.6_0.3'
features = np.load(f"{root}/{base_split}_feats.npy")
labels = np.load(f"{root}/{base_split}_labels.npy")
print(set(labels))
print(len(set(labels)))
ipdb.set_trace()

root = 'data/coco/any_shot_detection/base_det'
base_split = 'train_0.6_0.3'
features = np.load(f"{root}/{base_split}_feats.npy")
labels = np.load(f"{root}/{base_split}_labels.npy")

feat_save_path = f"{root}/{base_split}_feats_subset_for_quick_debug.npy"
label_save_path = f"{root}/{base_split}_labels_subset_for_quick_debug.npy"

features = features[:100, :]
labels = labels[:100]

np.save(feat_save_path, features)
np.save(label_save_path, labels)
ipdb.set_trace()
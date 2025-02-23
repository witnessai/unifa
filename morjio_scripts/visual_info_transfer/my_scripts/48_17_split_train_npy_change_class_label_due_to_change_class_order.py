import numpy as np 
import ipdb 
data_root = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/48_17_fsd_split/tfa/0shot'
trainsplit = 'old_train_0.6_0.3'
old_data = np.load(f"{data_root}/{trainsplit}_labels.npy", allow_pickle=True)  

old_order_names = ['person', 'bicycle', 'car', 'motorcycle', 'truck', 'boat', 'bench', 'bird', 'horse', 'sheep',
               'zebra', 'giraffe', 'backpack', 'handbag', 'skis', 'kite', 'surfboard', 'bottle', 'spoon',
               'bowl', 'banana', 'apple', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'chair', 'bed',
               'tv', 'laptop', 'remote', 'microwave', 'oven', 'refrigerator', 'book', 'clock', 'vase',
               'toothbrush', 'train', 'bear', 'suitcase', 'frisbee', 'fork', 'sandwich', 'toilet', 'mouse',
               'toaster', 'bg']
new_order_names = ['person', 'bicycle', 'car', 'motorcycle', 'train', 'truck', 'boat', 'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra', 'giraffe', 'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'kite', 'surfboard', 'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'chair', 'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'microwave', 'oven', 'toaster', 'refrigerator', 'book', 'clock', 'vase', 'toothbrush', 'bg']


## add shift
old_order_name2shiftlabelid = dict()
shift = 1000
for label_id, class_name in enumerate(old_order_names):
    old_data[old_data==label_id] = label_id+shift
    old_order_name2shiftlabelid[class_name] = label_id+shift
new_order_name2labelid = dict()
for label_id, class_name in enumerate(new_order_names):
    old_id = old_order_name2shiftlabelid[class_name]
    old_data[old_data==old_id] = label_id
    new_order_name2labelid[class_name] = label_id

print(old_order_name2shiftlabelid)
print(new_order_name2labelid)
ipdb.set_trace()
new_data = old_data
trainsplit = 'train_0.6_0.3'
np.save(f"{data_root}/{trainsplit}_labels.npy", new_data)
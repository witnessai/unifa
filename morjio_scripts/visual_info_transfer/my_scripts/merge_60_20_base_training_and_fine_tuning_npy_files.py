import numpy as np
import ipdb 

######################## 60/20 classes split ########################
ALL_CLASSES=('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
                 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
                 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                 'scissors', 'teddy bear', 'hair drier', 'toothbrush')
BASE_CLASSES=('truck', 'traffic light', 'fire hydrant', 'stop sign',
                  'parking meter', 'bench', 'elephant', 'bear', 'zebra',
                  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                  'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                  'kite', 'baseball bat', 'baseball glove', 'skateboard',
                  'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork',
                  'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                  'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                  'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote',
                  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                  'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                  'teddy bear', 'hair drier', 'toothbrush')
ids = np.where(np.isin(ALL_CLASSES, BASE_CLASSES))[0].tolist()
# print(ids)
# # the remaining classes are novel classes, print novel ids
# ids = np.where(np.isin(ALL_CLASSES, BASE_CLASSES, invert=True))[0].tolist()
# print(ids)
# ipdb.set_trace()
ids.sort(reverse=True)
# print(ids)


######################## 60/20 classes split ########################



######################## check test label id begin ########################
# label_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/10shot/test_0.6_0.3_labels.npy'
# label_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/30shot/test_0.6_0.3_labels.npy'
# labels = np.load(label_path)
# print(set(labels)) # set(labels) # {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 80}
# print(len(labels))
# ipdb.set_trace() 
######################## check test label id end ########################




######################## merge tfa 10shot begin ########################
## merge base traininig and fine tuning labels
# label_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/10shot/train_0.6_0.3_labels_base_training.npy'
# label_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/10shot/train_0.6_0.3_labels_fine_tuning.npy'
# label1 = np.load(label_path1) # set(label1) is [0, 60]
# label2 = np.load(label_path2) # set(label2) is [0, 80]
# ## only remap label1 
# base_num = 60
# all_num = 80
# label1[label1==base_num] = all_num # remap bg class
# for index, new_label_id in enumerate(ids): # remap fg classes
#     ori_label_id = base_num-index-1 # minus one for excloude bg class
#     # print(ori_label_id, new_label_id)
#     label1[label1==ori_label_id] = new_label_id

# feat_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/10shot/train_0.6_0.3_feats_base_training.npy'
# feat_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/10shot/train_0.6_0.3_feats_fine_tuning.npy'
# feat1 = np.load(feat_path1)
# feat2 = np.load(feat_path2)
# label = np.concatenate((label1, label2), axis=0)
# feat = np.concatenate((feat1, feat2), axis=0)
# save_label_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/10shot/train_0.6_0.3_labels.npy'
# save_feat_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/10shot/train_0.6_0.3_feats.npy'
# ipdb.set_trace()
# np.save(save_label_path, label)
# np.save(save_feat_path, feat)
######################## merge tfa 10shot end ########################


######################## merge tfa 30shot begin ########################
## merge base traininig and fine tuning labels
# label_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/30shot/train_0.6_0.3_labels_base_training.npy'
# label_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/30shot/train_0.6_0.3_labels_fine_tuning.npy'
# label1 = np.load(label_path1) # set(label1) is [0, 60]
# label2 = np.load(label_path2) # set(label2) is [0, 80]
# ## only remap label1 
# base_num = 60
# all_num = 80
# label1[label1==base_num] = all_num # remap bg class
# for index, new_label_id in enumerate(ids): # remap fg classes
#     ori_label_id = base_num-index-1 # minus one for excloude bg class
#     # print(ori_label_id, new_label_id)
#     label1[label1==ori_label_id] = new_label_id
# feat_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/30shot/train_0.6_0.3_feats_base_training.npy'
# feat_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/30shot/train_0.6_0.3_feats_fine_tuning.npy'
# feat1 = np.load(feat_path1)
# feat2 = np.load(feat_path2)
# label = np.concatenate((label1, label2), axis=0)
# feat = np.concatenate((feat1, feat2), axis=0)
# save_label_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/30shot/train_0.6_0.3_labels.npy'
# save_feat_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/30shot/train_0.6_0.3_feats.npy'
# ipdb.set_trace()
# np.save(save_label_path, label)
# np.save(save_feat_path, feat)
######################## merge tfa 30shot end ########################


######################## merge fsce 10shot begin ########################
## merge base traininig and fine tuning labels
# label_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/10shot/train_0.6_0.3_labels_base_training.npy'
# label_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/10shot/train_0.6_0.3_labels_fine_tuning.npy'
# label1 = np.load(label_path1) # set(label1) is [0, 60]
# label2 = np.load(label_path2) # set(label2) is [0, 80]
# ## only remap label1 
# base_num = 60
# all_num = 80
# label1[label1==base_num] = all_num # remap bg class
# for index, new_label_id in enumerate(ids): # remap fg classes
#     ori_label_id = base_num-index-1 # minus one for excloude bg class
#     # print(ori_label_id, new_label_id)
#     label1[label1==ori_label_id] = new_label_id

# feat_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/10shot/train_0.6_0.3_feats_base_training.npy'
# feat_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/10shot/train_0.6_0.3_feats_fine_tuning.npy'
# feat1 = np.load(feat_path1)
# feat2 = np.load(feat_path2)
# label = np.concatenate((label1, label2), axis=0)
# feat = np.concatenate((feat1, feat2), axis=0)
# save_label_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/10shot/train_0.6_0.3_labels.npy'
# save_feat_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/10shot/train_0.6_0.3_feats.npy'
# ipdb.set_trace()
# np.save(save_label_path, label)
# np.save(save_feat_path, feat)
# ######################## merge fsce 10shot end ########################




# ######################## merge fsce 30shot begin ########################
# ## merge base traininig and fine tuning labels
label_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/30shot/train_0.6_0.3_labels_base_training.npy'
label_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/30shot/train_0.6_0.3_labels_fine_tuning.npy'
label1 = np.load(label_path1) # set(label1) is [0, 60]
label2 = np.load(label_path2) # set(label2) is [0, 80]
## only remap label1 
base_num = 60
all_num = 80
label1[label1==base_num] = all_num # remap bg class
for index, new_label_id in enumerate(ids): # remap fg classes
    ori_label_id = base_num-index-1 # minus one for excloude bg class
    # print(ori_label_id, new_label_id)
    label1[label1==ori_label_id] = new_label_id

feat_path1 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/30shot/train_0.6_0.3_feats_base_training.npy'
feat_path2 = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/30shot/train_0.6_0.3_feats_fine_tuning.npy'
feat1 = np.load(feat_path1)
feat2 = np.load(feat_path2)
label = np.concatenate((label1, label2), axis=0)
feat = np.concatenate((feat1, feat2), axis=0)
save_label_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/30shot/train_0.6_0.3_labels.npy'
save_feat_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/30shot/train_0.6_0.3_feats.npy'
# ipdb.set_trace()
np.save(save_label_path, label)
np.save(save_feat_path, feat)
######################## merge fsce 30shot end ########################


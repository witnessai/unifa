import numpy as np
import ipdb 

src_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/48_17_fsd_split/tfa/0shot/test_0.6_0.3_labels_wrong_label_map.npy'

labels = np.load(src_path, allow_pickle=True)

seen_48_17 = [0, 1, 2, 3, 6, 7, 8, 9, 10, 13, 14, 17, 18, 19, 20, 22, 24, 25, 26, 28, 30, 31, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 48, 49, 50, 51, 52, 53, 55, 56, 57, 59, 60, 61, 62, 64, 65]
seen_48_17_continual2discrete = { idx:label for idx, label in enumerate(seen_48_17)}
continual2discrete = seen_48_17_continual2discrete

unseen_48_17 = [  4, 5, 11, 12, 15, 16, 21, 23, 27, 29, 32, 34, 45, 47, 54, 58, 63, 65]
unseen_48_17.sort(reverse=True)
unique_label = np.unique(labels).tolist()
unique_label.sort(reverse=True)
unseen_48_17_continual2discrete = {x:y for x, y in zip(unique_label, unseen_48_17)}
continual2discrete = unseen_48_17_continual2discrete



for iter_label in unique_label:
    labels[labels==iter_label] = continual2discrete[iter_label]
tgt_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/48_17_fsd_split/tfa/0shot/test_0.6_0.3_labels.npy'
np.save(tgt_path, labels)
ipdb.set_trace()
root_path = '/home/nieh/morjio/projects/detection/zero_shot_detection/RRFS/feats_data'
dir_list = ['48_17_coco_wrong_label_map', 
            '65_15_coco_suzsd_det_wrong_label_map', 
            '65_15_coco_wrong_label_map', 
            '65_15_coco_suzsd_det_increase_resolution_wrong_label_map']

# 0表示背景类，不做remap
seen_48_17 = [ 1,  2,  3,  4,  7,  8,  9, 10, 11, 14, 15, 18, 19, 20, 21, 23, 25,
       26, 27, 29, 31, 32, 34, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47,
       49, 50, 51, 52, 53, 54, 56, 57, 58, 60, 61, 62, 63, 65]
unseen_48_17 = [ 5,  6, 12, 13, 16, 17, 22, 24, 28, 30, 33, 35, 46, 48, 55, 59, 64]
seen_65_15 = [1,  2,  3,  4,  6,  8,  9, 10, 11, 12, 14, 15, 17, 18, 19, 20, 21,
       23, 24, 25, 26, 27, 28, 31, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
       44, 45, 46, 47, 48, 50, 51, 52, 54, 55, 56, 57, 58, 59, 60, 61, 63,
       64, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 80]
unseen_65_15 = [5,  7, 13, 16, 22, 29, 30, 32, 43, 49, 53, 62, 65, 71, 79]

seen_48_17_continual2discrete = { (idx+1):label for idx, label in enumerate(seen_48_17)}
unseen_48_17_continual2discrete = {(idx+1):label for idx, label in enumerate(unseen_48_17)}

seen_65_15_continual2discrete = {(idx+1):label for idx, label in enumerate(seen_65_15)}
unseen_65_15_continual2discrete = {(idx+1):label for idx, label in enumerate(unseen_65_15)}

for dir_path in dir_list:
    split_name_list = ['train_0.6_0.3', 'test_0.6_0.3']
    for split_name in split_name_list:
        labels = np.load(f"{root_path}/{dir_path}/{split_name}_labels.npy")
        unique_label_length = len(set(labels))
        if unique_label_length == 65+1:
            continual2discrete = seen_65_15_continual2discrete
        elif unique_label_length == 15+1:
            continual2discrete = unseen_65_15_continual2discrete
        elif unique_label_length == 48+1:
            continual2discrete = seen_48_17_continual2discrete
        elif unique_label_length == 17+1:
            continual2discrete = unseen_48_17_continual2discrete
        for iter_label in range(unique_label_length-1, 0, -1):
            labels[labels==iter_label] = continual2discrete[iter_label]
        new_dir_path = dir_path.replace('_wrong_label_map', '')
        save_path = f"{root_path}/{new_dir_path}/{split_name}_labels.npy"
        print(save_path)
        # ipdb.set_trace()
        np.save(save_path, labels)

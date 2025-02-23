import os 
import numpy as np
import ipdb 

root_dirs = [
    '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection/base_det/', 
    '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection/base_few_shot_det/', 
    '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection/base_few_shot_det/unseen15_feats_labels/', 
    '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection/base_few_shot_det/multi_labels/zero_shot/']

filenames = ['test_0.6_0.3_labels_without_remap.npy', 'train_0.6_0.3_labels_without_remap.npy']

# filenames = ['test_0.6_0.3_zero_shot_labels_without_remap.npy']

zero_shot_label_id_with_bg_65_8_7 = [4, 15, 28, 29, 48, 61, 64, 80]
zero_shot_map_65_8_7_for_base_det = {0:4, 1:15, 2:28, 3:29, 4:48, 5:61, 6:64, 65:80}
zero_shot_map_65_8_7_for_base_few_shot_det = {0:4, 1:15, 2:28, 3:29, 4:48, 5:61, 6:64, 80:80}
unseen_map_65_15_for_base_few_shot_det = {0:4, 1:6, 2:12, 3:15, 4:21, 5:28, 6:29, 7:31, 8:42, 9:48, 10:52, 11:61, 12:64, 13:70, 14:78, 80:80}

base_map_label_id_with_bg_65_8_7 = [ 0,  1,  2,  3,  5,  7,  8,  9, 10, 11, 13, 14, 16, 17, 18, 19, 20,
       22, 23, 24, 25, 26, 27, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
       43, 44, 45, 46, 47, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 62,
       63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 79, 80]
base_map_65_8_7_for_base_det = {idx:label for idx, label in enumerate(base_map_label_id_with_bg_65_8_7)}
# base_map_65_8_7_for_base_det: {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11, 10: 13, 11: 14, 12: 16, 13: 17, 14: 18, 15: 19, 16: 20, 17: 22, 18: 23, 19: 24, 20: 25, 21: 26, 22: 27, 23: 30, 24: 32, 25: 33, 26: 34, 27: 35, 28: 36, 29: 37, 30: 38, 31: 39, 32: 40, 33: 41, 34: 43, 35: 44, 36: 45, 37: 46, 38: 47, 39: 49, 40: 50, 41: 51, 42: 53, 43: 54, 44: 55, 45: 56, 46: 57, 47: 58, 48: 59, 49: 60, 50: 62, 51: 63, 52: 65, 53: 66, 54: 67, 55: 68, 56: 69, 57: 71, 58: 72, 59: 73, 60: 74, 61: 75, 62: 76, 63: 77, 64: 79, 65: 80}
base_map_65_8_7_for_base_few_shot_det = {idx:label for idx, label in enumerate(base_map_label_id_with_bg_65_8_7[:-1])}
base_map_65_8_7_for_base_few_shot_det[80] = 80
# base_map_65_8_7_for_base_few_shot_det: {0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 7, 6: 8, 7: 9, 8: 10, 9: 11, 10: 13, 11: 14, 12: 16, 13: 17, 14: 18, 15: 19, 16: 20, 17: 22, 18: 23, 19: 24, 20: 25, 21: 26, 22: 27, 23: 30, 24: 32, 25: 33, 26: 34, 27: 35, 28: 36, 29: 37, 30: 38, 31: 39, 32: 40, 33: 41, 34: 43, 35: 44, 36: 45, 37: 46, 38: 47, 39: 49, 40: 50, 41: 51, 42: 53, 43: 54, 44: 55, 45: 56, 46: 57, 47: 58, 48: 59, 49: 60, 50: 62, 51: 63, 52: 65, 53: 66, 54: 67, 55: 68, 56: 69, 57: 71, 58: 72, 59: 73, 60: 74, 61: 75, 62: 76, 63: 77, 64: 79, 80: 80}


for i, root_dir in enumerate(root_dirs):
    # if i == 0:
    #     zero_shot_map_65_8_7 = zero_shot_map_65_8_7_for_base_det
    #     base_map_65_8_7 = base_map_65_8_7_for_base_det
    # else:
    #     zero_shot_map_65_8_7 = zero_shot_map_65_8_7_for_base_few_shot_det
    #     base_map_65_8_7 = base_map_65_8_7_for_base_few_shot_det
    if i <= 2:
        continue 
    else:
        zero_shot_map_65_8_7 = zero_shot_map_65_8_7_for_base_few_shot_det# unseen_map_65_15_for_base_few_shot_det
        base_map_65_8_7 = base_map_65_8_7_for_base_few_shot_det
    ipdb.set_trace()
    for fn in filenames:
        file_path = os.path.join(root_dir, fn)
        if not os.path.exists(file_path):  continue 
        # 判断是否文件存在
        labels = np.load(file_path)
        new_labels = np.zeros(labels.shape[0], np.int32)
        if 'test' in fn:
            keys = zero_shot_map_65_8_7.keys()
            keys = list(keys)
            keys.sort(reverse = True)
            for key in keys:
                new_labels[labels==key] = zero_shot_map_65_8_7[key]
            save_name = 'test_0.6_0.3_labels.npy'
        elif 'train' in fn:
            keys = base_map_65_8_7.keys()
            keys = list(keys)
            keys.sort(reverse = True)
            for key in keys:
                new_labels[labels==key] = base_map_65_8_7[key]
            save_name = 'train_0.6_0.3_labels.npy'
        save_path = os.path.join(root_dir, save_name)
        np.save(save_path, new_labels)
# ipdb.set_trace()    
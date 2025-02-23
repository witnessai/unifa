import torch
import ipdb 
import numpy as np
import os 
# zero_shot_id = np.array([4, 15, 28, 29, 48, 61, 64])
# few_zero_shot_id = np.array([4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78])
# loc_of_zero_shot_in_few_zero_shot = np.where(np.isin(few_zero_shot_id, zero_shot_id))[0]
# # output: loc_of_zero_shot_in_few_zero_shot
# # [ 0,  3,  5,  6,  9, 11, 12]
# loc_of_zero_shot_in_few_zero_shot_bg = loc_of_zero_shot_in_few_zero_shot+1
# # [ 1,  4,  6,  7, 10, 12, 13]
# suzsd_unseen_classifier_path = '/home/niehui/morjio/projects/detection/zero-shot-detection/Synthesizing_the_Unseen_for_Zero-shot_Object_Detection/checkpoints/coco_65_15_morjio_v2_author_pretrained_model/classifier_best.pth'
# suzsd_unseen_classifier = torch.load(suzsd_unseen_classifier_path)

# # ipdb.set_trace()
# # 把背景类别的参数调整到最后
# suzsd_unseen_classifier['state_dict']['fc1.weight'][-1] = suzsd_unseen_classifier['state_dict']['fc1.weight'][0]
# suzsd_unseen_classifier['state_dict']['fc1.bias'][-1] = suzsd_unseen_classifier['state_dict']['fc1.bias'][0]
# for i, loc in enumerate(loc_of_zero_shot_in_few_zero_shot_bg):
#     suzsd_unseen_classifier['state_dict']['fc1.weight'][i] = suzsd_unseen_classifier['state_dict']['fc1.weight'][loc]
#     suzsd_unseen_classifier['state_dict']['fc1.bias'][i] = suzsd_unseen_classifier['state_dict']['fc1.bias'][loc]
# # 问题：1.bias没赋值，2.背景类的参数没有调到最后

# save_path = 'checkpoints/asd_65_8_7/classifier_best_from_suzsd_v3_fix_order_bg_bias.pth'
# torch.save(suzsd_unseen_classifier, save_path)
# ipdb.set_trace()



## 加载unseen 15 upper bound分类器

zero_shot_id = np.array([4, 15, 28, 29, 48, 61, 64, 80])

# output: loc_of_zero_shot_in_few_zero_shot
# [ 0,  3,  5,  6,  9, 11, 12]
unseen15_upper_bound_classifier_root  = 'checkpoints/asd_65_8_7/unseen15_classifier_upper_bound/train_2nd'
# unseen15_upper_bound_classifier_root  = 'checkpoints/asd_65_8_7/2022-05-19-few-zero-shot/'
unseen15_upper_bound_classifier_path = os.path.join(unseen15_upper_bound_classifier_root, 'classifier_best.pth')
# unseen15_upper_bound_classifier_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/asd_65_8_7/2022-05-11-base-few-zero-shot/classifier_best.pth'
# unseen15_upper_bound_classifier_path = 'checkpoints/asd_65_8_7/2022-05-18-base-few-zero-shot/classifier_best_reorder_zs_cls.pth'
# unseen15_upper_bound_classifier_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/asd_65_8_7/2022-05-09-few-zero-shot/classifier_best.pth'
unseen15_upper_bound_classifier = torch.load(unseen15_upper_bound_classifier_path)
cls_weight = unseen15_upper_bound_classifier['state_dict']['fc1.weight']
cls_bias = unseen15_upper_bound_classifier['state_dict']['fc1.bias']
if cls_bias.shape[0] == 81:
    few_zero_shot_id = np.arange(81)
elif cls_bias.shape[0] == 16:
    few_zero_shot_id = np.array([4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78, 80])

loc_of_zero_shot_in_few_zero_shot = np.where(np.isin(few_zero_shot_id, zero_shot_id))[0]
print(loc_of_zero_shot_in_few_zero_shot)
print(torch.mean(torch.abs(cls_weight)))

for i, loc in enumerate(loc_of_zero_shot_in_few_zero_shot):
    unseen15_upper_bound_classifier['state_dict']['fc1.weight'][i] = unseen15_upper_bound_classifier['state_dict']['fc1.weight'][loc]
    unseen15_upper_bound_classifier['state_dict']['fc1.bias'][i] = unseen15_upper_bound_classifier['state_dict']['fc1.bias'][loc]
print(unseen15_upper_bound_classifier['state_dict']['fc1.weight'][:8])
print(unseen15_upper_bound_classifier['state_dict']['fc1.bias'][:8])
# ipdb.set_trace()
# save_path = 'checkpoints/asd_65_8_7/2022-05-11-base-few-zero-shot/classifier_best_reorder_zs_cls.pth'
# save_path = 'checkpoints/asd_65_8_7/2022-05-09-few-zero-shot/classifier_best_reorder_zs_cls.pth'
save_path = os.path.join(unseen15_upper_bound_classifier_root, 'classifier_best_reorder_zs_cls.pth')
torch.save(unseen15_upper_bound_classifier, save_path)
ipdb.set_trace()
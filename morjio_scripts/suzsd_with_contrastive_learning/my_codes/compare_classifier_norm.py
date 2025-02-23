import torch  
import ipdb 

pretrained_det_path = 'checkpoints/asd_65_8_7/merged_det_model/merged_base_few_zero_shot_det_model_upper_bound_unseen15.pth'
pretrained_det = torch.load(pretrained_det_path)


cls_weight = pretrained_det['state_dict']['roi_head.bbox_head.fc_cls.weight']
cls_weight = torch.abs(cls_weight)

base_label_id = [ 0,  1,  2,  3,  5,  7,  8,  9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 79]
few_shot_label_id = [ 6, 12, 21, 31, 42, 52, 70, 78]
base_few_label_id = base_label_id + few_shot_label_id
base_few_label_id.sort()
zero_shot_label_id = [ 4, 15, 28, 29, 48, 61, 64]


base_few_norm = torch.mean(cls_weight[base_few_label_id])
zero_norm = torch.mean(cls_weight[zero_shot_label_id])
print(base_few_norm)
print(zero_norm)

# torch.mean(cls_weight[29])

ipdb.set_trace()


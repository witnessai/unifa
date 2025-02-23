import torch  
import numpy as np
import ipdb

cls_path = '/home/niehui/morjio/projects/detection/zero-shot-detection/Synthesizing_the_Unseen_for_Zero-shot_Object_Detection/checkpoints/coco_65_15_morjio_v2_author_pretrained_model/classifier_best.pth'


det_path = '/home/niehui/morjio/projects/detection/zero-shot-detection/Synthesizing_the_Unseen_for_Zero-shot_Object_Detection/mmdetection/work_dirs/author_pretrained_model/epoch_12.pth'


det_weight = torch.load(det_path)
cls_weight = torch.load(cls_path)


det_weight = det_weight['state_dict']['bbox_head.fc_cls.weight']
cls_weight = cls_weight['state_dict']['fc1.weight']
det_weight = torch.abs(det_weight)
cls_weight = torch.abs(cls_weight)


seen_label_id = [ 0,  1,  2,  3,  5,  7,  8,  9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 79]
ipdb.set_trace()
seen_norm = torch.mean(det_weight[seen_label_id])
unseen_norm = torch.mean(cls_weight[:])
print(seen_norm)
print(unseen_norm)


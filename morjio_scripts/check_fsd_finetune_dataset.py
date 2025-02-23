from pycocotools.coco import COCO
import ipdb 
import json 
import os

## 检查经典小样本训练集的实例数量
# annFile = 'data/few_shot_ann/coco/benchmark_10shot_all_class_trainval.json'
# annFile = 'data/any_shot_ann/coco/annotations/finetune_dataset_seed42/full_box_10shot_trainval_deduplication.json'

# annFile = 'data/few_shot_ann/coco/tfa/seed7_10shot_all_class_trainval.json'
# coco=COCO(annFile)
# fsd_finetune_dataset_imgId = coco.getImgIds()

# total_ann_num = 0
# annFile = 'data/coco/annotations/instances_train2017.json'
# coco=COCO(annFile)
# annId = coco.getAnnIds(imgIds=fsd_finetune_dataset_imgId)
# total_ann_num += len(annId)

# annFile = 'data/coco/annotations/instances_val2017.json'
# coco=COCO(annFile)
# annId = coco.getAnnIds(imgIds=fsd_finetune_dataset_imgId)
# total_ann_num += len(annId)

# print(total_ann_num) # check if len(annId) is equal to 799
# ipdb.set_trace()



## 提取经典小样本训练集中包含ASD few-shot类的imgid
ID2CLASS = { 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush",     }
few_shot_cat_ids = [7, 14, 23, 36, 48, 58, 80, 89]
novel_imgid_list = []
for cat_id in few_shot_cat_ids:
    class_name = ID2CLASS[cat_id]
    annFile = f'/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/few_shot_ann_60_20_split/coco/benchmark_10shot/full_box_10shot_{class_name}_trainval.json'
    coco=COCO(annFile)
    fsd_finetune_dataset_imgId = coco.getImgIds()
    novel_imgid_list.extend(fsd_finetune_dataset_imgId)

novel_class_imgid_file_name = 'novel_class_imgid_of_classic_fsd_finetune_dataset.txt'
with open(novel_class_imgid_file_name, 'w') as f:
    novel_imgid_list = [str(x)+'\n' for x in novel_imgid_list]
    f.writelines(novel_imgid_list)
ipdb.set_trace()    
import argparse
import json
import os
import random
import ipdb
from pycocotools.coco import COCO

seed = 42
random.seed(seed)

seen_cat_ids = [1, 2, 3, 4, 6, 8, 9, 10, 11, 13, 15, 16, 18, 19, 20, 21, 22, 24, 25, 27, 28, 31, 32, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 67, 72, 73, 75, 76, 77, 78, 79, 81, 82, 84, 85, 86, 87, 88, 90]
unseen_cat_ids = [5, 7, 14, 17, 23, 33, 34, 36, 48, 54, 58, 70, 74, 80, 89] # 65/15 few-shot
# 第二种小样本类别划分
few_shot_cat_ids = [7, 14, 17, 23, 36, 48, 70, 80] # 65/8 any-shot, 8 few-shot classes
zero_shot_cat_ids = list(set(unseen_cat_ids)-set(few_shot_cat_ids))
all_cat_ids = seen_cat_ids+few_shot_cat_ids
all_cat_ids.sort()

few_shot_ann_num = {i: 0 for i in few_shot_cat_ids}
all_ann_num = {i: 0 for i in all_cat_ids}


## 先从FSD经典微调训练集中挑选
ID2CLASS = { 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush",     }
novel_imgid_list = []
for cat_id in few_shot_cat_ids:
    class_name = ID2CLASS[cat_id]
    annFile = f'/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/few_shot_ann_60_20_split/coco/benchmark_10shot/full_box_10shot_{class_name}_trainval.json'
    coco=COCO(annFile)
    fsd_finetune_dataset_imgId = coco.getImgIds()
    novel_imgid_list.extend(fsd_finetune_dataset_imgId)
annFile = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/annotations/instances_train2014.json'
coco = COCO(annFile)
imgIds = coco.getImgIds()
intersection_imgIds = list(set(novel_imgid_list) &  set(imgIds) )
sample_imgIds = []
for imgid in intersection_imgIds:
    annIds = coco.getAnnIds(imgIds=imgid, iscrowd=None)
    anns = coco.loadAnns(ids=annIds)
    append_flag = True
    for ann in anns:
        cat_id = ann['category_id']
        if cat_id in zero_shot_cat_ids:
            append_flag = False
    if append_flag:
        sample_imgIds.append(imgid)
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id in few_shot_cat_ids:
                few_shot_ann_num[cat_id] += 1
            elif cat_id in all_cat_ids:
                all_ann_num[cat_id] += 1
print(len(sample_imgIds))
print(few_shot_ann_num)
print(all_ann_num)


## 再从train2014中挑选
shot_num = 10
max_instances_num_in_image = 3
novel_class_imgid_list = []
for idx, catId in enumerate(all_cat_ids):
    print(idx, catId)
    loop_num = 0
    while all_ann_num[catId] < shot_num:
        tmp_few_shot_num = {i: 0 for i in few_shot_cat_ids}
        tmp_all_num = {i: 0 for i in all_cat_ids}
        tmp_seen_num = {i: 0 for i in seen_cat_ids}
        imgIds = coco.getImgIds(catIds=catId)
        imgId = random.sample(imgIds, 1)
        annId = coco.getAnnIds(imgIds=imgId, iscrowd=None)
        anns = coco.loadAnns(ids=annId)
        append_flag = True
        instances_num_in_img = 0
        for ann in anns:
            if ann['category_id'] in few_shot_cat_ids:
                tmp_few_shot_num[ann['category_id']] += 1
                tmp_all_num[ann['category_id']] += 1
                instances_num_in_img += 1
            if ann['category_id'] in seen_cat_ids:
                tmp_seen_num[ann['category_id']] += 1
                tmp_all_num[ann['category_id']] += 1
                instances_num_in_img += 1
            if ann['category_id'] in zero_shot_cat_ids:
                append_flag = False
        for i in few_shot_cat_ids:
            if few_shot_ann_num[i] + tmp_few_shot_num[i] > shot_num:
                append_flag = False
        if instances_num_in_img>max_instances_num_in_image and loop_num<10000:
            append_flag = False
        else:
            append_flag = True
        if append_flag:
            save_imgid_flag = False
            if imgId[0] not in sample_imgIds:
                for i in few_shot_cat_ids:
                    if tmp_few_shot_num[i]>0:
                        save_imgid_flag = True
                    few_shot_ann_num[i] += tmp_few_shot_num[i]
                for i in all_cat_ids:
                    all_ann_num[i] += tmp_all_num[i]
                # print(imgId)
                if save_imgid_flag: novel_class_imgid_list.append(imgId[0])
                sample_imgIds.extend(imgId)
        else:
            loop_num += 1
print(all_ann_num)
print(len(all_ann_num))
annFile = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/annotations/instances_train2014.json'
data = json.load(open(annFile)) 
annId = coco.getAnnIds(imgIds=sample_imgIds, iscrowd=None)
sample_shots = coco.loadAnns(ids=annId)
sample_imgs = coco.loadImgs(ids=sample_imgIds)   
# sample_cats = []
# for cat in data['categories']:
#     if cat['id'] in all_cat_ids:
#         sample_cats.append(cat)
new_data = {
    'info': data['info'],
    'licenses': data['licenses'],
    'images': sample_imgs,
    'annotations': sample_shots,
    'categories': data['categories'],
}


# novel_class_imgid_file_path = f'/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/morjio_scripts/novel_class_imgid_of_rahman_asd_{shot_num}shot_finetune_dataset_{seed}_v2.txt'
# with open(novel_class_imgid_file_path, 'w') as f:
#     novel_class_imgid_list = [str(x)+'\n' for x in novel_class_imgid_list] 
#     f.writelines(novel_class_imgid_list)


ipdb.set_trace()
save_path = f'/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/any_shot_ann_65_8_7_split/rahman_asd_full_box_{shot_num}shot_trainval_{seed}_v4_for_fs_set2.json'
with open(save_path, 'w') as f:
    json.dump(new_data, f)



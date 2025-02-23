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
few_shot_cat_ids = [7, 14, 23, 36, 48, 58, 80, 89] # 65/8 any-shot, 8 few-shot classes
zero_shot_cat_ids = list(set(unseen_cat_ids)-set(few_shot_cat_ids))
all_cat_ids = seen_cat_ids+few_shot_cat_ids
all_cat_ids.sort()

few_shot_ann_num = {i: 0 for i in few_shot_cat_ids}
all_ann_num = {i: 0 for i in all_cat_ids}

annFile = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/annotations/instances_train2014.json'
# data = json.load(open(annFile))
coco = COCO(annFile)

sample_imgIds = []
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



save_path = f'/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/any_shot_ann_65_8_7_split/rahman_asd_full_box_{shot_num}shot_trainval_{seed}_v3.json'
with open(save_path, 'w') as f:
    json.dump(new_data, f)



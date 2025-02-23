import argparse
import json
import os
import random
import ipdb
from pycocotools.coco import COCO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43],
                        help="Range of seeds")
    parser.add_argument("--shot_nums", type=int,  nargs="+", default=[1],
                        help="shot number")
    args = parser.parse_args()
    return args

args = parse_args()
# ipdb.set_trace()
seeds = args.seeds
shot_nums = args.shot_nums
for shot_num in shot_nums:
    for seed in seeds:
        random.seed(seed)
        seen_cat_ids = [1, 2, 3, 4, 6, 8, 9, 10, 11, 13, 15, 16, 18, 19, 20, 21, 22, 24, 25, 27, 28, 31, 32, 35, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 55, 56, 57, 59, 60, 61, 62, 63, 64, 65, 67, 72, 73, 75, 76, 77, 78, 79, 81, 82, 84, 85, 86, 87, 88, 90]
        unseen_cat_ids = [5, 7, 14, 17, 23, 33, 34, 36, 48, 54, 58, 70, 74, 80, 89] # 65/15 few-shot
        # unseen_cat_ids = [7, 14, 23, 36, 48, 58, 80, 89] # 65/8 any-shot, 8 few-shot classes
        all_cat_ids = seen_cat_ids+unseen_cat_ids
        all_cat_ids.sort()

        unseen_ann_num = {i: 0 for i in unseen_cat_ids}
        all_ann_num = {i: 0 for i in all_cat_ids}

        annFile = 'data/coco/annotations/instances_train2014.json'
        # data = json.load(open(annFile))
        coco = COCO(annFile)

        sample_imgIds = []
        for catId in all_cat_ids:
            print(catId)
            while all_ann_num[catId] < shot_num:
                tmp_unseen_num = {i: 0 for i in unseen_cat_ids}
                tmp_all_num = {i: 0 for i in all_cat_ids}
                imgIds = coco.getImgIds(catIds=catId)
                imgId = random.sample(imgIds, 1)
                annId = coco.getAnnIds(imgIds=imgId, iscrowd=None)
                anns = coco.loadAnns(ids=annId)
                for ann in anns:
                    if ann['category_id'] in unseen_cat_ids:
                        tmp_unseen_num[ann['category_id']] += 1
                    if ann['category_id'] in all_cat_ids:
                        tmp_all_num[ann['category_id']] += 1
                append_flag = True
                for i in unseen_cat_ids:
                    if unseen_ann_num[i] + tmp_unseen_num[i] > shot_num:
                        append_flag = False
                if append_flag:
                    if imgId[0] not in sample_imgIds:
                        for i in unseen_cat_ids:
                            unseen_ann_num[i] += tmp_unseen_num[i]
                        for i in all_cat_ids:
                            all_ann_num[i] += tmp_all_num[i]
                        print(imgId)
                        sample_imgIds.extend(imgId)

        annFile = 'data/coco/annotations/instances_train2014.json'
        data = json.load(open(annFile)) 
        annId = coco.getAnnIds(imgIds=sample_imgIds, iscrowd=None)
        sample_shots = coco.loadAnns(ids=annId)
        sample_imgs = coco.loadImgs(ids=sample_imgIds)   
        new_data = {
            'info': data['info'],
            'licenses': data['licenses'],
            'images': sample_imgs,
            'annotations': sample_shots,
            'categories': data['categories'],
        }
        # save_path = f'data/any_shot_ann/coco/annotations/rahman_fsd_finetune_dataset/rahman_fsd_full_box_{shot_num}shot_trainval.json'
        save_path = f'data/few_shot_ann_65_15_split/rahman_fsd_full_box_{shot_num}shot_trainval_{seed}.json'
        with open(save_path, 'w') as f:
            json.dump(new_data, f)



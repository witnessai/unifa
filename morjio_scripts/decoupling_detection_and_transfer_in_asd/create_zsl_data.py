import os
import json
import ipdb
from pycocotools.coco import COCO 
import cv2
from tqdm import tqdm

# img_root = '/data1/niehui/MSCOCO/train2014'
# save_root = '/data1/niehui/decoupling_detection_and_transfer_in_asd/coco_bbox_crop_for_zsl/65_15_split/train'
# ann_file = '/data1/niehui/MSCOCO/annotations/instances_shafin_65_train.json'


img_root = '/data1/niehui/MSCOCO/val2014'
# save_root = '/data1/niehui/decoupling_detection_and_transfer_in_asd/coco_bbox_crop_for_zsl/65_15_split/zsl_test'
save_root = '/data1/niehui/decoupling_detection_and_transfer_in_asd/coco_bbox_crop_for_zsl/65_15_split/gzsl_test'
ann_file = '/data1/niehui/MSCOCO/annotations/instances_shafin_test.json'
coco_gt = COCO(ann_file)
img_infos = coco_gt.dataset['images']
anno = coco_gt.dataset['annotations']
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat', 'traffic light', 
                'fire hydrant', 'stop sign', 'bench', 'bird', 'dog', 'horse', 'sheep', 'cow', 
                'elephant', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'skis', 
                'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
                'tennis racket', 'bottle', 'wine glass', 'cup', 'knife', 'spoon', 'bowl', 'banana', 
                'apple', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'cake', 'chair', 'couch',
                 'potted plant', 'bed', 'dining table', 'tv', 'laptop', 'remote', 'keyboard', 
                 'cell phone', 'microwave', 'oven', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                  'scissors', 'teddy bear', 'toothbrush', 'airplane', 'train', 'parking meter', 'cat', 'bear', 'suitcase', 'frisbee', 'snowboard', 'fork', 'sandwich', 'hot dog', 'toilet', 'mouse', 'toaster', 'hair drier')

cat2label = {1: 1, 2: 2, 3: 3, 4: 4, 6: 5, 8: 6, 9: 7, 10: 8, 11: 9, 13: 10, 15: 11, 16: 12, 18: 13, 19: 14, 20: 15, 21: 16, 22: 17, 24: 18, 25: 19, 27: 20, 28: 21, 31: 22, 32: 23, 35: 24, 37: 25, 38: 26, 39: 27, 40: 28, 41: 29, 42: 30, 43: 31, 44: 32, 46: 33, 47: 34, 49: 35, 50: 36, 51: 37, 52: 38, 53: 39, 55: 40, 56: 41, 57: 42, 59: 43, 60: 44, 61: 45, 62: 46, 63: 47, 64: 48, 65: 49, 67: 50, 72: 51, 73: 52, 75: 53, 76: 54, 77: 55, 78: 56, 79: 57, 81: 58, 82: 59, 84: 60, 85: 61, 86: 62, 87: 63, 88: 64, 90: 65, 5: 66, 7: 67, 14: 68, 17: 69, 23: 70, 33: 71, 34: 72, 36: 73, 48: 74, 54: 75, 58: 76, 70: 77, 74: 78, 80: 79, 89: 80}


seen_label = [ 0,  1,  2,  3,  5,  7,  8,  9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 79]

# img_names = []ls
min_size = 32
iter_num = 0
for img_idx, img_info in enumerate(tqdm(img_infos)):
    img_name = img_info['file_name']# %%
    img_path = os.path.join(img_root, img_name)
    # print(img_path)
    img = cv2.imread(img_path)
    # img_names.append(img_name)

    image_id = img_info['id']
    ann_ids = coco_gt.getAnnIds(image_id)
    anns = coco_gt.loadAnns(ann_ids)
    for i, ann in enumerate(anns):
        bbox = ann['bbox']
        bbox = [int(x) for x in bbox]
        x, y, w, h = bbox
        if min(w, h) < min_size: # filter small object in train dataset
            continue
        category = ann['category_id']
        label_index = cat2label[category]
        
        # for zsl_test
        if '/zsl_test' in save_root:
            if label_index in seen_label:
                continue

        # label = CLASSES[label_index-1]
        crop_img = img[y:y+h, x:x+w]
        save_name = img_name.split('.')[0] + '_' + str(i) + '_' + str(label_index) + '.jpg'

        save_path = os.path.join(save_root, save_name)
        # ipdb.set_trace()
        cv2.imwrite(save_path, crop_img)


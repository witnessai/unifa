import json 
import os
import ipdb
from pycocotools.coco import COCO


src_path = '/data1/niehui/MSCOCO/annotations/instances_val2014.json'

# coco = COCO(src_path)
with open(src_path, 'r') as f:
    data = json.load(f)

new_data = dict()
new_data['info'] = data['info']
new_data['licenses'] = data['licenses']
new_data['categories'] = data['categories']

target_img = ['COCO_val2014_000000020650.jpg', 'COCO_val2014_000000027897.jpg']
imginfo_list = []
imgid_list = []
for img in data['images']:
    if img['file_name'] in target_img:
        print(img['file_name'])
        imginfo_list.append(img)
        imgid_list.append(img['id'])

anno_list = []
for anno in data['annotations']:
    if anno['image_id'] in imgid_list:
        anno_list.append(anno)

new_data['annotations'] = anno_list
new_data['images'] = imginfo_list

save_path = '/data1/niehui/MSCOCO/annotations/instances_val2014_select_img_for_vis.json'
with open(save_path, 'w') as f:
    json.dump(new_data, f)

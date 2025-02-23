import os 
import json 
import ipdb 
from lvis import LVIS, LVISVis

# check 65/15 coco train and test 
# coco_65_15_train = '/data1/niehui/MSCOCO/annotations/instances_shafin_65_train.json'
# coco_65_15_test = '/data1/niehui/MSCOCO/annotations/instances_shafin_test.json'
# with open(coco_65_15_train) as fa, open(coco_65_15_test) as fb:
#     data_a = json.load(fa)
#     data_b = json.load(fb)
#     print(len(data_a['categories'])) # 65
#     print(len(data_b['categories'])) # 79
#     ipdb.set_trace()

data_type = 'train'
data_type = 'val'
anno_path = f'/data1/niehui/lvis/annotations/lvis_v1_{data_type}.json'
with open(anno_path) as fd:
    dict_data = json.load(fd)

data = LVIS(anno_path)
cats = data.load_cats(ids=None)
id2freq = dict()
for cat in cats:
    id2freq[cat['id']] = cat['frequency']

img_ids = data.get_img_ids()

exclude_img_ids = []
for img_id in img_ids:
    # img_info = data.load_imgs([img_id])
    ann_ids = data.get_ann_ids(img_ids=[img_id])
    has_rare = False
    for ann in ann_ids:
        ann_info = data.load_anns([ann])
        if id2freq[ann_info[0]['category_id']] == 'r':
            has_rare = True
    if has_rare is True:
        exclude_img_ids.append(img_id)
print(len(img_ids)) # 100170 for train
print(len(exclude_img_ids)) # 1462 for train



# check whether class number will decrease after removing exclude_img_ids from img_ids
img_ids_after_remove_rare = list(set(img_ids) - set(exclude_img_ids))
classid_after_remove_rare = []
for img_id in img_ids_after_remove_rare:
    ann_ids = data.get_ann_ids(img_ids=[img_id])
    for ann in ann_ids:
        ann_info = data.load_anns([ann])
        if ann_info[0]['category_id'] not in classid_after_remove_rare:
            classid_after_remove_rare.append(ann_info[0]['category_id'])
print(len(classid_after_remove_rare)) # 866 for train, 


imgs_after_remove_rare = data.load_imgs(ids=img_ids_after_remove_rare)
ann_ids_after_remove_rare = data.get_ann_ids(img_ids=img_ids_after_remove_rare)
anns_after_remove_rare = data.load_anns(ann_ids_after_remove_rare)
cats_after_remove_rare = data.load_cats(ids=classid_after_remove_rare)

new_data = dict()
new_data['info'] = dict_data['info']
new_data['licenses'] = dict_data['licenses']
new_data['categories'] = cats_after_remove_rare
new_data['images'] = imgs_after_remove_rare
new_data['annotations'] = anns_after_remove_rare

new_anno_path = f'/data1/niehui/lvis/annotations/lvis_v1_{data_type}_remove_rare.json'
with open(new_anno_path, 'w') as fd:
    json.dump(new_data, fd)

ipdb.set_trace()
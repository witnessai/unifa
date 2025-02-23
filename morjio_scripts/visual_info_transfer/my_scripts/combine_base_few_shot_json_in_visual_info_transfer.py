import os
import ipdb 
import json 

base_json = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/annotations/instances_train2014_seen_65_15.json'
shot_num = 5

# few_json = f'/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/few_shot_ann_65_15_split/rahman_fsd_only_cover_novel_full_box_10shot_trainval.json'
few_json = f'/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/few_shot_ann_65_15_split/rahman_fsd_full_box_{shot_num}shot_trainval_42.json' 

with open(base_json) as f1, open(few_json) as f2:
    base_json = json.load(f1)
    few_json = json.load(f2)

combine_json = dict()
combine_json['info'] = base_json['info']
combine_json['licenses'] = base_json['licenses']
combine_json['categories'] = few_json['categories']


register_img_id = []
register_anno_id = []
for item in base_json['images']:
    register_img_id.append(item['id'])
for item in base_json['annotations']:
    register_anno_id.append(item['id'])

filter_duplicate_in_few_json_img = []
filter_duplicate_in_few_json_anno = []
for item in few_json['images']:
    if item['id'] not in register_img_id:
        register_img_id.append(item['id'])
        filter_duplicate_in_few_json_img.append(item)
for item in few_json['annotations']:
    if item['id'] not in register_anno_id:
        register_anno_id.append(item['id'])
        filter_duplicate_in_few_json_anno.append(item)
print(len(filter_duplicate_in_few_json_img), len(filter_duplicate_in_few_json_anno))
# ipdb.set_trace()
combine_json['images'] = base_json['images'] + filter_duplicate_in_few_json_img
combine_json['annotations'] = base_json['annotations'] + filter_duplicate_in_few_json_anno

# base_img_id_list, few_img_id_list = [], []
# for item in base_json['images']:
#     base_img_id_list.append(item['id'])
# for item in few_json['images']:
#     few_img_id_list.append(item['id'])
# few_anno_id_list = []
# for item in few_json['annotations']:
#     few_anno_id_list.append(item['id'])
# print(len(base_img_id_list))
# print(len(set(base_img_id_list)))
# print(len(few_img_id_list))
# print(len(set(few_img_id_list)))
# print(len(few_anno_id_list))
# print(len(set(few_anno_id_list)))
# print(set(base_img_id_list) & set(few_img_id_list))
# output is:
# 61598
# 61598
# 114
# 112
# 738
# 715

# ipdb.set_trace()




new_json_name = f'base_few_shot_json_for_extract_feats_in_visual_info_transfer_{shot_num}shot.json'
save_path = os.path.join('/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/annotations/combine_base_few_shot_json_in_visual_info_transfer', new_json_name)
with open(save_path, 'w') as fd:
    json.dump(combine_json, fd)
ipdb.set_trace()




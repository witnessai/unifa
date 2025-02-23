import os 
import json
import ipdb 

# root_dir = '/home/niehui/morjio/projects/detection/any_shot_detection/mmfewshot/data/any_shot_ann/coco/annotations/finetune_dataset_seed42/10shot'
# root_dir = 'data/few_shot_ann/coco/benchmark_10shot'
root_dir = 'data/few_shot_ann/coco/tfa/seed7'
file_list = os.listdir(root_dir)


new_data = dict()
set_info_licenses_categories_flag = False
imgid_list = []
annoid_list = []

for file_name in file_list:
    if 'full_box_10shot' not in file_name: continue
    file_path = os.path.join(root_dir, file_name)
    with open(file_path) as fd:
        data = json.load(fd)
        # print(data.keys()) # dict_keys(['info', 'licenses', 'images', 'annotations', 'categories'])
        if not set_info_licenses_categories_flag:
            new_data = data 
            set_info_licenses_categories_flag = True
            for img in new_data['images']:
                imgid_list.append(img['id'])
            for ann in new_data['annotations']:
                annoid_list.append(ann['id'])
        else:
            for img in data['images']:
                if img['id'] in imgid_list:
                    continue
                else:
                    imgid_list.append(img['id'])
                    new_data['images'].append(img)
            for ann in data['annotations']:
                if ann['image_id'] not in imgid_list: ipdb.set_trace()
                if ann['id'] in annoid_list:
                    continue
                else:
                    annoid_list.append(ann['id'])
                    new_data['annotations'].append(ann) 
            # new_data['images'].extend(data['images'])
            # new_data['annotations'].extend(data['annotations'])
# save_name = 'data/any_shot_ann/coco/annotations/finetune_dataset_seed42/full_box_10shot_trainval_deduplication.json'
# save_name = 'data/few_shot_ann/coco/benchmark_10shot_all_class_trainval.json'
save_name = 'data/few_shot_ann/coco/tfa/seed7_10shot_all_class_trainval.json'
with open(save_name, 'w') as fd:
    json.dump(new_data, fd)



from pycocotools.coco import COCO 
import os
import ipdb 
import json

####COCO Dataset#########################################################################################################
datatype = 'val' # datatype = 'train'
src_cocodata_path = '/home/niehui/morjio/data/datasets/MSCOCO/annotations/instances_{}2014.json'.format(datatype)
coco = COCO(src_cocodata_path)
coco_select_classnames = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat',
                'traffic light', 'fire hydrant', 'stop sign', 'bench', 'bird', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                'skis', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
                'tennis racket', 'bottle', 'wine glass', 'cup', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'orange', 'broccoli', 'carrot',  'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'tv', 'laptop','remote', 'keyboard', 'cell phone', 'microwave',
                'oven', 'sink', 'refrigerator', 'book', 'clock',   'vase', 'scissors', 'teddy bear', 'toothbrush']
coco_select_catid = coco.getCatIds(catNms=coco_select_classnames)
coco_select_imgid = []
for catid in coco_select_catid:
    tmp_imgid = coco.getImgIds(imgIds=[], catIds=[catid])
    coco_select_imgid.extend(tmp_imgid)
coco_select_imgid = list(set(coco_select_imgid))

save_imgids = []
annotations = []
for imgid in coco_select_imgid:
    ann_ids = coco.getAnnIds(imgIds=[imgid])
    anns = coco.loadAnns(ids=ann_ids)
    save = True
    for ann in anns:
        if ann['category_id'] not in coco_select_catid:
            save = False
            break
    if save:
        save_imgids.append(imgid)
        annotations.extend(anns)
images = coco.loadImgs(ids=save_imgids)
categories = coco.loadCats(ids=coco_select_catid)

new_json = dict()
new_json['info'] = dict()
new_json['licenses'] = []
new_json['images'] = images
new_json['annotations'] = annotations
new_json['categories'] = categories

save_root_dir = '/home/niehui/morjio/projects/detection/any_shot_detection/mmfewshot/data/any_shot_ann/coco/annotations/'
new_json_name = 'instances_{}2014_base_65_15.json'.format(datatype)
save_path = os.path.join(save_root_dir, new_json_name)
with open(save_path, 'w') as fd:
    json.dump(new_json, fd)
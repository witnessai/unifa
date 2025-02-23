import json 
import ipdb 
from pycocotools.coco import COCO


annfile = 'data/few_shot_ann/coco/annotations/val.json'
with open(annfile) as fd:
    data = json.load(fd)

coco = COCO(annfile)

new_data = dict()
new_data['info'] = data['info']
new_data['licenses'] = data['licenses']
new_data['categories'] = data['categories']
new_data['images'] = data['images'][:10]
new_img_ids = []
for img_info in new_data['images']:
    new_img_ids.append(img_info['id'])

annIds = coco.getAnnIds(imgIds=new_img_ids)
anns = coco.loadAnns(annIds)
new_data['annotations'] = anns


ipdb.set_trace()

with open('data/few_shot_ann/coco/annotations/val_subset_10imgs_for_quick_debug.json', 'w') as fd:
    json.dump(new_data, fd)
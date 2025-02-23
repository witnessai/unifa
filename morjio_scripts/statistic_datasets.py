from pycocotools.coco import COCO
import ipdb 
import json 
import os

dataDir='/path/to/your/cocoDataset'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# annFile = 'data/few_shot_ann/coco/annotations/val.json'
# annFile = 'data/coco/annotations/instances_val2014.json'
# annFile = 'data/any_shot_ann/coco/annotations/finetune_dataset_seed42/full_box_10shot_trainval.json'
# annFile = 'data/any_shot_ann/coco/annotations/finetune_dataset_seed42/full_box_10shot_trainval_deduplication.json'
# annFile = 'data/few_shot_ann/coco/benchmark_10shot_all_class_trainval.json'
# annFile = 'data/few_shot_ann/coco/annotations/trainvalno5k.json'
# annFile = 'data/few_shot_ann/coco/tfa/seed1_10shot_all_class_trainval.json'
# annFile = 'data/any_shot_ann/coco/annotations/rahman_fsd_finetune_dataset/rahman_fsd_full_box_10shot_trainval.json'
# annFile = 'data/coco/annotations/instances_shafin_test_morjio_for_asd.json'
annFile = 'data/any_shot_ann/coco/annotations/rahman_asd_finetune_dataset/rahman_asd_full_box_10shot_trainval.json'
# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
cat_nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(cat_nms)))

cat_ids = coco.getCatIds()
# 统计各类的图片数量和标注框数量
imgId_list = []
annId_list = []
for i, catId in enumerate(cat_ids):
    imgId = coco.getImgIds(catIds=catId)
    annId = coco.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)
    imgId_list.extend(imgId)
    annId_list.extend(annId)
    print("{:<15} {:<6d}     {:<10d}".format(cat_nms[i], len(imgId), len(annId)))
print('total images number is %d ' % len(set(imgId_list))) # 统计有注释的图片数量
print('total annotations number is %d ' % len(set(annId_list)))

# 没注释的图片数量统计
# annFile = 'data/any_shot_ann/coco/annotations/finetune_dataset_seed42/full_box_10shot_trainval_deduplication.json'
# annFile = 'data/few_shot_ann/coco/benchmark_10shot_all_class_trainval.json'
# annFile = 'data/few_shot_ann/coco/annotations/trainvalno5k.json'
# annFile = 'data/few_shot_ann/coco/tfa/seed1_10shot_all_class_trainval.json'
# annFile = 'data/any_shot_ann/coco/annotations/rahman_fsd_finetune_dataset/rahman_fsd_full_box_10shot_trainval.json'
# annFile = 'data/coco/annotations/instances_shafin_test_morjio_for_asd.json'
# annFile = 'data/few_shot_ann/coco/annotations/val.json'
# annFile = 'data/coco/annotations/instances_val2014.json'
annFile = 'data/any_shot_ann/coco/annotations/rahman_asd_finetune_dataset/rahman_asd_full_box_10shot_trainval.json'
with open(annFile) as fd:
    data = json.load(fd)
print(len(data['images']))
print(len(data['annotations']))
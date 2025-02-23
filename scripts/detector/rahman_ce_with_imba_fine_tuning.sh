# python tools/detection/train.py configs/detection/asd/coco/rahman_ce_with_imba_asd_r101_fpn_coco_10shot-fine-tuning.py

# ./tools/detection/dist_train.sh configs/detection/asd/coco/rahman_ce_with_imba_asd_r101_fpn_coco_10shot-fine-tuning.py 3


## rahman_ce_with_imba_fsd_65_15_sem_r101_fpn_coco_10shot-fine-tuning
# python tools/detection/train.py configs/detection/asd/coco/rahman_ce_with_imba_fsd_65_15_sem_r101_fpn_coco_10shot-fine-tuning.py


## rahman_ce_with_imba_retinanet_fsd_65_15_sem_r101_fpn_coco_10shot-fine-tuning
CUDA_VISIBLE_DEVICES=3 python tools/detection/train.py configs/detection/asd/coco/rahman_ce_with_imba_retinanet_fsd_65_15_sem_r101_fpn_coco_10shot-fine-tuning.py


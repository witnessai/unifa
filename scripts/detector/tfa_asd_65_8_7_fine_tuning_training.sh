## multi-gpus
# ./tools/detection/dist_train.sh configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning.py 4 


## single gpu
# CUDA_VISIBLE_DEVICES=5 python tools/detection/train.py configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning.py  

## single gpu, add seesaw loss
# CUDA_VISIBLE_DEVICES=4 python tools/detection/train.py configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_add_seesawloss.py  

## multi-gpus, add seesaw loss
# CUDA_VISIBLE_DEVICES=2,3,4,5 ./tools/detection/dist_train.sh configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_add_seesawloss.py 4 


## add gdl of DeFRCN
# CUDA_VISIBLE_DEVICES=4 python tools/detection/train.py configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_add_gdl.py  

## add gdl of DeFRCN without freeze
# CUDA_VISIBLE_DEVICES=5 python tools/detection/train.py configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_add_gdl_without_freeze.py  

## fs set 2
# CUDA_VISIBLE_DEVICES=5 python tools/detection/train.py configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set2.py  

## fs set 3
# CUDA_VISIBLE_DEVICES=4 python tools/detection/train.py configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3.py  

## fs set 4
# CUDA_VISIBLE_DEVICES=5 python tools/detection/train.py configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set4.py  


## fs set 3, 5shot
CUDA_VISIBLE_DEVICES=4 python tools/detection/train.py configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3_5shot.py 

## fs set 3, 1shot
# CUDA_VISIBLE_DEVICES=5 python tools/detection/train.py configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3_1shot.py 
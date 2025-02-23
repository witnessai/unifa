# multiple gpus
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/detection/dist_train.sh configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_3shot-fine-tuning.py 4

# single gpu
CUDA_VISIBLE_DEVICES=4 python tools/detection/train.py configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_3shot-fine-tuning.py  

# multi-gpus
CUDA_VISIBLE_DEVICES=2,3,4,5 ./tools/detection/dist_train.sh configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_base-training.py 4

# multi-gpus
# ./tools/detection/dist_train.sh configs/detection/asd/coco/asd_r50_fpn_coco_base-training.py 3

# single gpu
# CUDA_VISIBLE_DEVICES=2 python tools/detection/train.py configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_base-training.py

# single gpu
# python tools/detection/train.py configs/detection/asd/coco/asd_r101_fpn_coco_base-training.py  



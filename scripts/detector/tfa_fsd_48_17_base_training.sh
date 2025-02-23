# multi-gpus
CUDA_VISIBLE_DEVICES=2,3,4,5 ./tools/detection/dist_train.sh configs/detection/asd/coco/fsd_48_17_r101_fpn_coco_base-training.py 4



# single gpu
# CUDA_VISIBLE_DEVICES=2 python tools/detection/train.py configs/detection/asd/coco/fsd_48_17_r101_fpn_coco_base-training.py




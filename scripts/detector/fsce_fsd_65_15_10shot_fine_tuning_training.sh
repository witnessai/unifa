# multi gpus
# ./tools/detection/dist_train.sh configs/detection/asd/coco/fsd_65_15_fsce_r101_fpn_coco_10shot-fine-tuning.py 4

# single gpu
CUDA_VISIBLE_DEVICES=0 python tools/detection/train.py configs/detection/asd/coco/fsd_65_15_fsce_r101_fpn_coco_10shot-fine-tuning.py
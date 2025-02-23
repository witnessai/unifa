# multi gpus
# ./tools/detection/dist_train.sh configs/detection/asd/coco/fsd_65_15_fsce_r101_fpn_coco_1shot-fine-tuning.py 4

# single gpu
# CUDA_VISIBLE_DEVICES=1 python tools/detection/train.py configs/detection/asd/coco/fsd_65_15_fsce_r101_fpn_coco_1shot-fine-tuning.py


# FSCE 只保留baseline改进，去掉对比学习模块
CUDA_VISIBLE_DEVICES=3 python tools/detection/train.py configs/detection/asd/coco/fsd_65_15_fsce_r101_fpn_coco_1shot-fine-tuning_wo_contra_loss.py
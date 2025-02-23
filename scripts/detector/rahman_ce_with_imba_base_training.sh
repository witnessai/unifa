## single gpu
# CUDA_VISIBLE_DEVICES=3 python tools/detection/train.py configs/detection/asd/coco/rahman_ce_with_imba_retinanet_fsd_65_15_sem_r101_fpn_coco_base-training.py


## multi gpus
./tools/detection/dist_train.sh configs/detection/asd/coco/rahman_ce_with_imba_retinanet_fsd_65_15_sem_r101_fpn_coco_base-training.py 4
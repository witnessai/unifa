## multi-gpus
#./tools/detection/dist_train.sh configs/detection/asd/coco/asd_sem_r101_fpn_coco_base-training.py 3 

## multi-gpus
./tools/detection/dist_train.sh configs/detection/asd/coco/asd_sem_r50_fpn_coco_base-training.py 3 

## single gpu
# python tools/detection/train.py configs/detection/asd/coco/asd_sem_r101_fpn_coco_base-training.py  
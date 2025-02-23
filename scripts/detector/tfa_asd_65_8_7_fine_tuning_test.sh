## multi-gpus
# ./tools/detection/dist_test.sh configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning.py 4 


## single gpu
CUDA_VISIBLE_DEVICES=3 python tools/detection/test.py configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning.py  work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning/iter_160000_ft_data_v4.pth --eval bbox
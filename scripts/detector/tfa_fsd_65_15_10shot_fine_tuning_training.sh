# multiple gpus
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/detection/dist_train.sh configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_10shot-fine-tuning.py 4

# single gpu
# CUDA_VISIBLE_DEVICES=2 python tools/detection/train.py configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_10shot-fine-tuning.py  


## attention generator
CUDA_VISIBLE_DEVICES=1 python tools/detection/train.py configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_attention_generator.py  
# CUDA_VISIBLE_DEVICES=1,2,3,5 ./tools/detection/dist_train.sh configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_attention_generator.py  4
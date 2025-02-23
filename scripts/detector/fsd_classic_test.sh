## tfa 10shot 
# CUDA_VISIBLE_DEVICES=2 python ./tools/detection/test.py \
#   configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_60_20_10shot_finetune_det_classifier.py \
#   work_dirs/tfa_r101_fpn_coco_10shot-fine-tuning_34server/iter_160000.pth --eval bbox



## tfa 30shot 
# CUDA_VISIBLE_DEVICES=0 python ./tools/detection/test.py \
#   configs/detection/tfa/coco/tfa_r101_fpn_coco_30shot-fine-tuning_visual_info_transfer_60_20_30shot_finetune_det_classifier.py \
#   work_dirs/tfa_r101_fpn_coco_30shot-fine-tuning_34server/iter_240000.pth --eval bbox


## fsce 10shot 
# CUDA_VISIBLE_DEVICES=0 python ./tools/detection/test.py \
#   configs/detection/fsce/coco/fsce_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_60_20_10shot_finetune_det_classifier.py \
#   work_dirs/fsce_r101_fpn_coco_10shot-fine-tuning_34server/iter_30000.pth --eval bbox


## fsce 30shot 
# CUDA_VISIBLE_DEVICES=1 python ./tools/detection/test.py \
#   configs/detection/fsce/coco/fsce_r101_fpn_coco_30shot-fine-tuning_visual_info_transfer_60_20_30shot_finetune_det_classifier.py \
#   work_dirs/fsce_r101_fpn_coco_30shot-fine-tuning_34server/iter_40000.pth --eval bbox
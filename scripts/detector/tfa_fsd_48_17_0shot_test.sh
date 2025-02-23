## gzsd test
# CUDA_VISIBLE_DEVICES=1 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_48_17_r101_fpn_coco_0shot_visual_info_transfer_two_softmax_from_suzsd_gzsd.py \
#     work_dirs/fsd_48_17_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --eval bbox

## zsd test
# CUDA_VISIBLE_DEVICES=2 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_48_17_r101_fpn_coco_0shot_visual_info_transfer_two_softmax_from_suzsd_zsd.py \
#     work_dirs/fsd_48_17_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --eval bbox


## gzsd test, v2(48 and 65) 
# CUDA_VISIBLE_DEVICES=1 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_48_17_r101_fpn_coco_0shot_visual_info_transfer_two_softmax_from_suzsd_gzsd_48_65.py \
#     work_dirs/fsd_48_17_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --eval bbox


# qwen,gzsd test, v2(48 and 65) 
# CUDA_VISIBLE_DEVICES=1 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_48_17_r101_fpn_coco_0shot_visual_info_transfer_two_softmax_from_suzsd_gzsd_48_65_qwen.py \
#     work_dirs/fsd_48_17_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --eval bbox


## zsd test
CUDA_VISIBLE_DEVICES=2 python ./tools/detection/test.py \
    configs/detection/asd/coco/fsd_48_17_r101_fpn_coco_0shot_visual_info_transfer_two_softmax_from_suzsd_zsd_qwen.py \
    work_dirs/fsd_48_17_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --eval bbox
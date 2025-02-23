## two softmax from suzsd, zsd
# CUDA_VISIBLE_DEVICES=3 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_1shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_zsd.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --eval bbox
## two softmax from suzsd, gzsd, 65+80
# CUDA_VISIBLE_DEVICES=3 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_1shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_gzsd_65_80.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --eval bbox


## two softmax from suzsd, gzsd, 65+80
CUDA_VISIBLE_DEVICES=3 python ./tools/detection/test.py \
    configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_1shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_gzsd_65_80_qwen.py \
    work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --eval bbox
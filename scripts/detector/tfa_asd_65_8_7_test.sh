## gasd协议评测，two softmax from suzsd
# CUDA_VISIBLE_DEVICES=3 python ./tools/detection/test.py \
#     configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_gasd.py \
#     work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3/base_model_random_init_bbox_head_for_fs_set3.pth --eval bbox
## asd协议评测，two softmax from suzsd
# CUDA_VISIBLE_DEVICES=3 python ./tools/detection/test.py \
#     configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_asd.py \
#     work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3/base_model_random_init_bbox_head_for_fs_set3.pth --eval bbox


## gasd协议评测，two softmax from suzsd, v2(73 and 15)
# CUDA_VISIBLE_DEVICES=3 python ./tools/detection/test.py \
#     configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_gasd_73_15.py \
#     work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3/base_model_random_init_bbox_head_for_fs_set3.pth --eval bbox
## asd协议评测，two softmax from suzsd, v2(73 and 15)
# 10shot
# CUDA_VISIBLE_DEVICES=5 python ./tools/detection/test.py \
#     configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_asd_73_15.py \
#     work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3/base_model_random_init_bbox_head_for_fs_set3.pth --eval bbox
# 5shot
# CUDA_VISIBLE_DEVICES=3 python ./tools/detection/test.py \
#     configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_asd_73_15.py \
#     work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3_5shot/base_model_random_init_bbox_head_for_fs_set3.pth --eval bbox
# 1shot
# CUDA_VISIBLE_DEVICES=4 python ./tools/detection/test.py \
#     configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_asd_73_15.py \
#     work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3_1shot/base_model_random_init_bbox_head_for_fs_set3.pth --eval bbox


## gasd协议评测，two softmax from suzsd, v3(73 and 80)
# 10shot
CUDA_VISIBLE_DEVICES=5 python ./tools/detection/test.py \
    configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_gasd_73_80.py \
    work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3/base_model_random_init_bbox_head_for_fs_set3.pth --eval bbox --show-dir inference_results/asd_65_8_7
# 5shot
# CUDA_VISIBLE_DEVICES=1 python ./tools/detection/test.py \
#     configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_gasd_73_80.py \
#     work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3_5shot/base_model_random_init_bbox_head_for_fs_set3.pth --eval bbox
# 1shot
# CUDA_VISIBLE_DEVICES=2 python ./tools/detection/test.py \
#     configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_gasd_73_80.py \
#     work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3_1shot/base_model_random_init_bbox_head_for_fs_set3.pth --eval bbox
## 微调分类器的可视化结果，目前看起来是比较好的
# CUDA_VISIBLE_DEVICES=1 python tools/detection/test.py configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_finetune_det_classifier.py  work_dirs/fsd_65_15_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth --eval bbox  --eval-options classwise=True --show-dir inference_results/fsd_65_15

## 直接融合概率的可视化结果
# CUDA_VISIBLE_DEVICES=1 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_direct_probabilities.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth --eval bbox  --eval-options classwise=True --show-dir inference_results/fsd_65_15


# CUDA_VISIBLE_DEVICES=2 python ./tools/detection/test.py \
    # configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_1shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd.py \
    # work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --eval bbox  --show-dir inference_results/fsd_65_15

# 2023/05/24 det all
# CUDA_VISIBLE_DEVICES=5 python ./tools/detection/test.py \
    # configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_gasd_73_80.py \
    # work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3/base_model_random_init_bbox_head_for_fs_set3.pth --eval bbox --show-dir inference_results/asd_65_8_7/


# det samples
CUDA_VISIBLE_DEVICES=5 python ./tools/detection/test.py \
    configs/detection/asd/coco/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_gasd_73_80_for_vis.py \
    work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3/base_model_random_init_bbox_head_for_fs_set3.pth --eval bbox --show-dir inference_results/samples

# CUDA_VISIBLE_DEVICES=5 python ./tools/detection/asod_inference.py 
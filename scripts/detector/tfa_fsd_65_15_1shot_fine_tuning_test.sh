# CUDA_VISIBLE_DEVICES=0 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_1shot-fine-tuning.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_1shot-fine-tuning/iter_16000.pth --eval bbox




## 测试小样本分类器在生成特征上微调，把微调后的FS分类器安回去
# CUDA_VISIBLE_DEVICES=1 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_1shot-fine-tuning_visual_info_transfer_finetune_det_classifier.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_1shot-fine-tuning/iter_16000.pth --eval bbox

## 测试检测81分类器在生成特征上微调，把微调后的81分类器和原来分类器输出进行融合
# CUDA_VISIBLE_DEVICES=5 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_1shot-fine-tuning_visual_info_transfer_finetune_det_classifier_add_base.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_1shot-fine-tuning/iter_160000.pth --eval bbox

## 测试小样本分类器，以取最大值的方式融合两个分类器概率值probabilities
# CUDA_VISIBLE_DEVICES=1 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_1shot-fine-tuning_visual_info_transfer_direct_probabilities.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_1shot-fine-tuning/iter_160000.pth --eval bbox

## 测试小样本分类器，对两个分类器单独做softmax，然后再合并
# CUDA_VISIBLE_DEVICES=0 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_1shot-fine-tuning_visual_info_transfer_individual_softmax.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_1shot-fine-tuning/iter_160000.pth --eval bbox

## 测试小样本分类器，two softmax from suzsd
# CUDA_VISIBLE_DEVICES=1 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_1shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_1shot-fine-tuning/iter_16000.pth --eval bbox
# CUDA_VISIBLE_DEVICES=2 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_1shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_zsd.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --eval bbox
## 测试小样本分类器，two softmax from suzsd, gzsd
# CUDA_VISIBLE_DEVICES=2 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_1shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_gzsd.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --eval bbox
## 测试小样本分类器，two softmax from suzsd, gzsd, 1shot
# CUDA_VISIBLE_DEVICES=3 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_1shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_gzsd.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_1shot-fine-tuning/iter_16000.pth --eval bbox
## 测试小样本分类器，two softmax from suzsd, gzsd, x-shot
CUDA_VISIBLE_DEVICES=5 python ./tools/detection/test.py \
    configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_1shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_gzsd.py \
    work_dirs/fsd_65_15_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth --eval bbox
## 测试小样本分类器，two softmax from suzsd, zsd
# CUDA_VISIBLE_DEVICES=3 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_1shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_zsd.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --eval bbox
## 测试小样本分类器，two softmax from suzsd, gzsd, 65+80
# CUDA_VISIBLE_DEVICES=3 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_1shot-fine-tuning_visual_info_transfer_two_softmax_from_suzsd_gzsd_65_80.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --eval bbox
# CUDA_VISIBLE_DEVICES=3 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_fsce_r101_fpn_coco_3shot-fine-tuning.py \
#     work_dirs/fsd_65_15_fsce_r101_fpn_coco_3shot-fine-tuning/iter_30000.pth --eval bbox

# 测试小样本分类器在生成特征上微调，把微调后的FS分类器安回去
CUDA_VISIBLE_DEVICES=3 python ./tools/detection/test.py \
    configs/detection/asd/coco/fsd_65_15_fsce_r101_fpn_coco_3shot-fine-tuning_visual_info_transfer_finetune_det_classifier.py \
    work_dirs/fsd_65_15_fsce_r101_fpn_coco_3shot-fine-tuning/iter_30000.pth --eval bbox
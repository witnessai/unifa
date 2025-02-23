# CUDA_VISIBLE_DEVICES=3 python ./tools/detection/test.py \
#   configs/detection/asd/coco/fsd_65_15_fsce_r101_fpn_coco_10shot-fine-tuning.py \
#   work_dirs/asd_fsce_r101_fpn_coco_10shot-fine-tuning/iter_30000.pth --eval bbox
# note: original config file is configs/detection/asd/coco/asd_fsce_r101_fpn_coco_10shot-fine-tuning.py

# CUDA_VISIBLE_DEVICES=0 python ./tools/detection/test.py \
#   configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_10shot-fine-tuning.py \
#   work_dirs/asd_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth --eval bbox
# note: original config file is configs/detection/asd/coco/asd_r101_fpn_coco_10shot-fine-tuning.py

# CUDA_VISIBLE_DEVICES=0 python ./tools/detection/test.py \
#   configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_10shot-fine-tuning.py \
#   work_dirs/asd_r101_fpn_coco_10shot-fine-tuning_seed42/iter_160000.pth --eval bbox
# note: original config file is configs/detection/asd/coco/asd_r101_fpn_coco_10shot-fine-tuning.py

# CUDA_VISIBLE_DEVICES=3 python ./tools/detection/test.py \
#   configs/detection/asd/coco/fsd_65_15_sem_r101_fpn_coco_10shot-fine-tuning.py \
#   work_dirs/asd_sem_r101_fpn_coco_10shot-fine-tuning_seed42/iter_160000.pth --eval bbox
# note: original config file is configs/detection/asd/coco/asd_sem_r101_fpn_coco_10shot-fine-tuning.py

# test the reproduced Faster R-CNN+FPN CrossEntropy with imba loss(loss apply in all classes) in rahman fine tuning dataset, 32w iters
# CUDA_VISIBLE_DEVICES=4 python ./tools/detection/test.py \
#   configs/detection/asd/coco/rahman_ce_with_imba_fsd_65_15_r101_fpn_coco_10shot-fine-tuning.py \
#   work_dirs/rahman_ce_with_imba_fsd_65_15_r101_fpn_coco_10shot-fine-tuning/iter_320000.pth --eval bbox

# test the reproduced Faster R-CNN+FPN CrossEntropy with imba loss(loss apply in all classes) in rahman fine tuning dataset, 16w iters
# CUDA_VISIBLE_DEVICES=0 python ./tools/detection/test.py \
#   configs/detection/asd/coco/rahman_ce_with_imba_fsd_65_15_r101_fpn_coco_10shot-fine-tuning.py \
#   work_dirs/rahman_ce_with_imba_fsd_65_15_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth --eval bbox

# test the reproduced Faster R-CNN+FPN CrossEntropy in rahman fine tuning dataset, asd 65-8-7 split
# CUDA_VISIBLE_DEVICES=4 python ./tools/detection/test.py \
#   configs/detection/asd/coco/rahman_ce_asd_65_8_7_r101_fpn_coco_10shot-fine-tuning.py \
#   work_dirs/rahman_ce_asd_65_8_7_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth --eval bbox

# test the reproduced Faster R-CNN+FPN CrossEntropy in rahman fine tuning dataset
# CUDA_VISIBLE_DEVICES=3 python ./tools/detection/test.py \
#   configs/detection/asd/coco/rahman_ce_fsd_65_15_r101_fpn_coco_10shot-fine-tuning.py \
#   work_dirs/rahman_ce_fsd_65_15_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth --eval bbox


# 测试复现的Faster R-CNN+FPN CrossEntropy in rahman fine tuning dataset 加上零样本分类器的结果
# checkpoints/asd_65_8_7/merged_det_model/merged_base_few_zero_shot_det_model.pth
# export PATH=/home/niehui/anaconda2/pkgs/cudatoolkit-9.0-h13b8566_0/lib/:$PATH && export LD_LIBRARY_PATH=/home/niehui/anaconda2/pkgs/cudatoolkit-9.0-h13b8566_0/lib/:$LD_LIBRARY_PATH && \
# unseen15_classifier_upper_bound/classifier_best_reorder_zs_cls.pth

# 零小基类分类器合并
# CUDA_VISIBLE_DEVICES=0 python ./tools/detection/test.py \
#   configs/detection/asd/coco/rahman_ce_asd_65_8_7_r101_fpn_coco_10shot-fine-tuning.py \
#   work_dirs/rahman_ce_asd_65_8_7_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth  \
#   --eval bbox \
#   --syn_weights checkpoints/asd_65_8_7/2022-05-19-few-zero-shot/classifier_best_reorder_zs_cls.pth  \
#   --gasd \
#   --show-dir inference_results/

# 零小样本类分类器合并
# CUDA_VISIBLE_DEVICES=0 python ./tools/detection/test.py \
#   configs/detection/asd/coco/rahman_ce_asd_65_8_7_r101_fpn_coco_10shot-fine-tuning.py \
#   work_dirs/rahman_ce_asd_65_8_7_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth  \
#   --eval bbox \
#   --syn_weights checkpoints/asd_65_8_7/2022-05-09-few-zero-shot/classifier_best_reorder_zs_cls.pth  \
#   --gasd \
#   --show-dir inference_results/

# 零样本类分类器合并
# CUDA_VISIBLE_DEVICES=1 python ./tools/detection/test.py \
#   configs/detection/asd/coco/rahman_ce_asd_65_8_7_r101_fpn_coco_10shot-fine-tuning.py \
#   work_dirs/rahman_ce_asd_65_8_7_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth  \
#   --eval bbox \
#   --syn_weights checkpoints/asd_65_8_7/2022-06-09-zero-shot-cegzsl-wrongrealize/classifier_best.pth  \
#   --gasd \
  # --show-dir inference_results/



# --syn_weights checkpoints/asd_65_8_7/2022-05-11-base-few-zero-shot/classifier_best_reorder_zs_cls.pth 

# --syn_weights checkpoints/asd_65_8_7/2022-05-11-zero-shot/classifier_best.pth 



# CUDA_VISIBLE_DEVICES=1 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_10shot-fine-tuning.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth --eval bbox


## 测试小样本分类器融合， 直接融合logits
# CUDA_VISIBLE_DEVICES=0 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_direct_logits.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth --eval bbox


## 测试小样本分类器融合，attention generator
# CUDA_VISIBLE_DEVICES=0 python ./tools/detection/test.py \
# configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_attention_generator.py \
# work_dirs/fsd_65_15_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_attention_generator/iter_40000.pth --eval bbox


## 测试小样本分类器在生成特征上微调，把微调后的FS分类器安回去
CUDA_VISIBLE_DEVICES=0 python ./tools/detection/test.py \
    configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_finetune_det_classifier.py \
    work_dirs/fsd_65_15_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth --eval bbox

## 测试检测81分类器在生成特征上微调，把微调后的81分类器和原来分类器输出进行融合
# CUDA_VISIBLE_DEVICES=5 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_finetune_det_classifier_add_base.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth --eval bbox

## 测试小样本分类器，以取最大值的方式融合两个分类器概率值probabilities
# CUDA_VISIBLE_DEVICES=1 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_direct_probabilities.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth --eval bbox

## 测试小样本分类器，对两个分类器单独做softmax，然后再合并
# CUDA_VISIBLE_DEVICES=0 python ./tools/detection/test.py \
#     configs/detection/asd/coco/fsd_65_15_r101_fpn_coco_10shot-fine-tuning_visual_info_transfer_individual_softmax.py \
#     work_dirs/fsd_65_15_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth --eval bbox
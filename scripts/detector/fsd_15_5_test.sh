# test for "base training"  model
# CUDA_VISIBLE_DEVICES=0 python tools/detection/test.py configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_base-training.py work_dirs/tfa_r101_fpn_voc-split1_base-training/iter_18000.pth  --eval mAP

## fsce add cpe, 15/5, 3shot, 测试小样本分类器微调，然后再把分类器安回去
# CUDA_VISIBLE_DEVICES=3 python ./tools/detection/test.py \
#     configs/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_3shot-fine-tuning_visual_info_transfer_finetune_det_classifier.py \
#     work_dirs/fsce_r101_fpn_contrastive-loss_voc-split1_3shot-fine-tuning/iter_8000.pth --eval mAP

## fsce add cpe, 15/5, 5shot, 测试小样本分类器微调，然后再把分类器安回去
# CUDA_VISIBLE_DEVICES=4 python ./tools/detection/test.py \
#     configs/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_5shot-fine-tuning_visual_info_transfer_finetune_det_classifier.py \
#     work_dirs/fsce_r101_fpn_contrastive-loss_voc-split1_5shot-fine-tuning/iter_9000.pth --eval mAP

# fsce, 15/5, 3shot, 测试小样本分类器微调，然后再把分类器安回去
# CUDA_VISIBLE_DEVICES=1 python ./tools/detection/test.py \
#     configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_3shot-fine-tuning_visual_info_transfer_finetune_det_classifier.py \
#     work_dirs/fsce_r101_fpn_voc-split1_3shot-fine-tuning/iter_8000.pth --eval mAP

## fsce, 15/5, 5shot, 测试小样本分类器微调，然后再把分类器安回去
CUDA_VISIBLE_DEVICES=5 python ./tools/detection/test.py \
    configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_5shot-fine-tuning_visual_info_transfer_finetune_det_classifier.py \
    work_dirs/fsce_r101_fpn_voc-split1_5shot-fine-tuning/iter_9000.pth --eval mAP


## tfa, 15/5, 3shot, 测试小样本分类器微调，然后再把分类器安回去
# CUDA_VISIBLE_DEVICES=5 python ./tools/detection/test.py \
#     configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_3shot-fine-tuning_visual_info_transfer_finetune_det_classifier.py \
#     work_dirs/tfa_r101_fpn_voc-split1_3shot-fine-tuning/iter_12000.pth --eval mAP


## tfa, 15/5, 5shot, 测试小样本分类器微调，然后再把分类器安回去
# CUDA_VISIBLE_DEVICES=2 python ./tools/detection/test.py \
#     configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_5shot-fine-tuning_visual_info_transfer_finetune_det_classifier.py \
#     work_dirs/tfa_r101_fpn_voc-split1_5shot-fine-tuning/iter_20000.pth --eval mAP
# multi-gpus
# CUDA_VISIBLE_DEVICES=0,2,3,5 ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_3shot-fine-tuning.py 4



# tfa, split1, 3shot
# CUDA_VISIBLE_DEVICES=5 python tools/detection/train.py configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_3shot-fine-tuning.py

# tfa, split1, 5shot
# CUDA_VISIBLE_DEVICES=3 python tools/detection/train.py configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_5shot-fine-tuning.py


# fsce, split1, 3shot
# CUDA_VISIBLE_DEVICES=1 python tools/detection/train.py configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_3shot-fine-tuning.py


# fsce, split1, 5shot
# CUDA_VISIBLE_DEVICES=2 python tools/detection/train.py configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_5shot-fine-tuning.py


# fsce, split1, 3shot, add CPE loss
# CUDA_VISIBLE_DEVICES=1 python tools/detection/train.py configs/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_3shot-fine-tuning.py

# fsce, split1, 5shot, add CPE loss
CUDA_VISIBLE_DEVICES=2 python tools/detection/train.py configs/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_5shot-fine-tuning.py
###### FSCE
CUDA_VISIBLE_DEVICES=0,1,2,3 ./tools/detection/dist_train.sh configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_base-training.py 4

###### TFA
# multi-gpus
# CUDA_VISIBLE_DEVICES=0,2,3,5 ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_base-training.py 4

# single gpu,  split1
# CUDA_VISIBLE_DEVICES=5 python tools/detection/train.py configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_base-training.py

# single gpu, split2
# CUDA_VISIBLE_DEVICES=3 python tools/detection/train.py configs/detection/tfa/voc/split2/tfa_r101_fpn_voc-split2_base-training.py

# single gpu, split3
# CUDA_VISIBLE_DEVICES=2 python tools/detection/train.py configs/detection/tfa/voc/split3/tfa_r101_fpn_voc-split3_base-training.py



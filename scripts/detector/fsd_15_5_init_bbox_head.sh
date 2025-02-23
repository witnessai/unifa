## fsce, init head
python -m tools.detection.misc.initialize_bbox_head \
    --src1 work_dirs/fsce_r101_fpn_voc-split1_base-training/iter_18000.pth \
    --method random_init \
    --save-dir work_dirs/fsce_r101_fpn_voc-split1_base-training 

## tfa, init head
# python -m tools.detection.misc.initialize_bbox_head_15_5 \
#     --src1 work_dirs/tfa_r101_fpn_voc-split1_base-training/iter_18000.pth \
#     --method random_init \
#     --save-dir work_dirs/tfa_r101_fpn_voc-split1_base-training 
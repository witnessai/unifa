python -m tools.detection.misc.fsd_65_15_initialize_bbox_head \
    --src1 work_dirs/fsd_65_15_r101_fpn_coco_base-training/iter_110000.pth \
    --method random_init \
    --save-dir work_dirs/fsd_65_15_r101_fpn_coco_base-training \
    --coco


# python -m tools.detection.misc.initialize_bbox_head \
#     --src1 work_dirs/tfa_r101_fpn_voc-split1_base-training/iter_18000.pth \
#     --method random_init \
#     --save-dir work_dirs/tfa_r101_fpn_voc-split1_base-training 


# python -m tools.detection.misc.initialize_bbox_head \
#     --src1 work_dirs/tfa_r101_fpn_coco_base-training/iter_110000.pth \
#     --method random_init \
#     --save-dir work_dirs/tfa_r101_fpn_coco_base-training \
#     --coco 
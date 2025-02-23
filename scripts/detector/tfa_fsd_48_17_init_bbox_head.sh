python -m tools.detection.misc.initialize_bbox_head_48_17 \
    --src1 work_dirs/fsd_48_17_r101_fpn_coco_base-training/iter_110000.pth \
    --method random_init \
    --save-dir work_dirs/fsd_48_17_r101_fpn_coco_base-training \
    --coco
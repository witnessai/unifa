python -m tools.detection.misc.asd_sem_initialize_bbox_head \
    --src1 work_dirs/asd_sem_r101_fpn_coco_base-training/iter_110000_trainvaldiff_3gpus.pth \
    --method random_init \
    --save-dir work_dirs/asd_sem_r101_fpn_coco_base-training \
    --coco
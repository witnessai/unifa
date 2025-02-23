## 65_8 init head
# python -m tools.detection.misc.asd_65_8_initialize_bbox_head \
#     --src1 work_dirs/asd_65_8_7_r101_fpn_coco_base-training/iter_110000.pth \
#     --method random_init \
#     --save-dir work_dirs/asd_65_8_7_r101_fpn_coco_base-training \
#     --coco --fs_set 1
# python -m tools.detection.misc.asd_65_8_initialize_bbox_head \
#     --src1 work_dirs/asd_65_8_7_r101_fpn_coco_base-training/iter_110000.pth \
#     --method random_init \
#     --save-dir work_dirs/asd_65_8_7_r101_fpn_coco_base-training \
#     --coco --fs_set 2
# python -m tools.detection.misc.asd_65_8_initialize_bbox_head \
#     --src1 work_dirs/asd_65_8_7_r101_fpn_coco_base-training/iter_110000.pth \
#     --method random_init \
#     --save-dir work_dirs/asd_65_8_7_r101_fpn_coco_base-training \
#     --coco --fs_set 3
# python -m tools.detection.misc.asd_65_8_initialize_bbox_head \
#     --src1 work_dirs/asd_65_8_7_r101_fpn_coco_base-training/iter_110000.pth \
#     --method random_init \
#     --save-dir work_dirs/asd_65_8_7_r101_fpn_coco_base-training \
#     --coco --fs_set 4


## 73_7 init head
# 10shot
# python -m tools.detection.misc.asd_73_7_initialize_bbox_head \
#     --src1 work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set1/iter_160000_ft_data_v4.pth \
#     --method random_init \
#     --save-dir work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set1 \
#     --coco --fs_set 1
# python -m tools.detection.misc.asd_73_7_initialize_bbox_head \
#     --src1 work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set2/iter_160000.pth \
#     --method random_init \
#     --save-dir work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set2 \
#     --coco --fs_set 2
# python -m tools.detection.misc.asd_73_7_initialize_bbox_head \
#     --src1 work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3/iter_160000.pth \
#     --method random_init \
#     --save-dir work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3 \
#     --coco --fs_set 3
# 5shot, fs_set 3
python -m tools.detection.misc.asd_73_7_initialize_bbox_head \
    --src1 work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3_5shot/iter_35000.pth \
    --method random_init \
    --save-dir work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3_5shot \
    --coco --fs_set 3
# 1shot, fs_set 3
python -m tools.detection.misc.asd_73_7_initialize_bbox_head \
    --src1 work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3_1shot/iter_10000.pth \
    --method random_init \
    --save-dir work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3_1shot \
    --coco --fs_set 3
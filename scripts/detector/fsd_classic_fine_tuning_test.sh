# CUDA_VISIBLE_DEVICES=1 python ./tools/detection/test.py \
#   configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning.py \
#   work_dirs/tfa_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth --eval bbox



# with dpif
# CUDA_VISIBLE_DEVICES=1 python ./tools/detection/test.py configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning_with_dpif.py  work_dirs/tfa_r101_fpn_coco_10shot-fine-tuning_with_dpif/iter_160000.pth --eval bbox

# with dpif add asso loss inference fusion
CUDA_VISIBLE_DEVICES=1 python ./tools/detection/test.py configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning_with_dpif_add_asso_loss_inference_fusion.py  work_dirs/tfa_r101_fpn_coco_10shot-fine-tuning_with_dpif_add_asso_loss_inference_fusion/iter_160000.pth --eval bbox
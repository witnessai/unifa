# python tools/detection/train.py configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning.py

# with unit transfer
# CUDA_VISIBLE_DEVICES=0 python tools/detection/train.py configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning_with_unit_transfer.py

# with unit transfer only vis
# CUDA_VISIBLE_DEVICES=1 python tools/detection/train.py configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning_with_unit_transfer_only_vis.py

# with unit transfer only sem
# CUDA_VISIBLE_DEVICES=2 python tools/detection/train.py configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning_with_unit_transfer_only_sem.py

# with unit transfer debug
# CUDA_VISIBLE_DEVICES=3 python tools/detection/train.py configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning_with_unit_transfer.py


# with unit transfer plus reg
# CUDA_VISIBLE_DEVICES=3 python tools/detection/train.py configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning_with_unit_transfer_plus_reg.py

# with unit transfer add ft
# CUDA_VISIBLE_DEVICES=0 python tools/detection/train.py configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning_with_unit_transfer_add_ft.py

# with dpif
# CUDA_VISIBLE_DEVICES=1 python tools/detection/train.py configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning_with_dpif.py

# with dpif add asso loss
CUDA_VISIBLE_DEVICES=3 python tools/detection/train.py configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning_with_dpif_add_asso_loss.py

# with dpif add asso loss inference fusion
# CUDA_VISIBLE_DEVICES=4 python tools/detection/train.py configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning_with_dpif_add_asso_loss_inference_fusion.py
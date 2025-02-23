#  ./tools/detection/dist_train.sh configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning.py 3

# finetune  voc 
# CUDA_VISIBLE_DEVICES=0,1,3 ./tools/detection/dist_train.sh configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_5shot-fine-tuning.py 3

# finetune coco with unit transfer
./tools/detection/dist_train.sh configs/detection/tfa/coco/tfa_r101_fpn_coco_10shot-fine-tuning_with_unit_transfer.py 4
# multi-gpus
CUDA_VISIBLE_DEVICES=1,3,4,5 ./tools/detection/dist_train.sh configs/detection/asd/lvis/zsd_866_337_r101_fpn.py 4



# single gpu
# CUDA_VISIBLE_DEVICES=1 python tools/detection/train.py configs/detection/asd/lvis/zsd_866_337_r101_fpn.py

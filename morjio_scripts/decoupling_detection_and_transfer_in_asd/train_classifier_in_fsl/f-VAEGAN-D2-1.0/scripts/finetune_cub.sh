CUDA_VISIBLE_DEVICES=1 python finetune.py --cuda --nepoch 50 --lr 0.01 --dataset CUB --dataroot /BS/xian18/work/BGM/data/images/ --batch_size 64 --saveto resnet101_cub.pth.tar 

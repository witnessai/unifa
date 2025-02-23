from arguments import parse_args
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import numpy as np
from dataset import FeaturesCls
from train_cls import TrainCls
from splits import get_asd_zero_shot_class_ids
import ipdb 

randomseed = 42
random.seed(randomseed)
torch.manual_seed(randomseed)
torch.cuda.manual_seed_all(randomseed) 


opt = parse_args()

trainCls = TrainCls(opt)

start_epoch = 0
n_epoch = 100
# zero_shot_class_ids = get_asd_zero_shot_class_ids(opt.dataset, split=opt.classes_split)
# classes_to_train = np.concatenate((zero_shot_class_ids, [80]))
features = np.load(f"{opt.dataroot}/{opt.testsplit}_feats.npy")
labels = np.load(f"{opt.dataroot}/{opt.testsplit}_labels.npy")

# dataset = FeaturesCls(opt, split='test', classes_to_train=None)
# dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, num_workers=4, shuffle=True, pin_memory=True)
fg_num = 250
bg_num = fg_num * 2
for epoch in range(start_epoch, n_epoch):
    print("Epoch: %d" % epoch)
    in_feat = features
    in_label = labels
    unique_label = list(set(in_label))
    for iter, l in enumerate(unique_label):
        feat_cls_num = in_label[in_label==l].shape[0]
        if l != 80:
            select_num = fg_num
        else:
            select_num = bg_num
        select_idx = np.arange(feat_cls_num)
        np.random.shuffle(select_idx)
        select_idx = select_idx[:select_num]
        if iter == 0:
            train_feat = in_feat[in_label==l][select_idx]
            train_label = in_label[in_label==l][:select_num]
        else:
            train_feat = np.concatenate((train_feat, in_feat[in_label==l][select_idx]))
            train_label = np.concatenate((train_label, in_label[in_label==l][:select_num]))
    # ipdb.set_trace()
    trainCls(train_feat, train_label, gan_epoch=epoch)

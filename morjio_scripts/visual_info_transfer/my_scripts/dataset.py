import numpy as np
import torch
# import pandas as pd 
from torch.utils.data import Dataset

import os.path
from os import path
import ipdb 

class FeaturesCls(Dataset):
     
    def __init__(self, features=None, labels=None, val=False, split='seen', classes_to_train=None):
        self.root = 'data/coco/any_shot_detection/base_det'
        self.testsplit = None
        self.opt = None
        self.classes_to_train = classes_to_train
        self.classid_tolabels = None
        self.features = features
        self.labels = labels
        if self.classes_to_train is not None:
            self.classid_tolabels = {label: i for i, label in enumerate(self.classes_to_train)}
        print(f"class ids for unseen classifier {self.classes_to_train}")
        if 'test' in split:
            self.loadRealFeats(syn_feature=features, syn_label=labels, split=split)

    def loadRealFeats(self, syn_feature=None, syn_label=None, split='train'):
        # 当只想训练6类的零样本分类器时，加载的确实是zero_shot_0.6_0.3_xxx.npy数据
        if 'test' in split:
            self.features = np.load(f"{self.root}/{self.testsplit}_feats.npy")
            self.labels = np.load(f"{self.root}/{self.testsplit}_labels.npy")
            print(f"{len(self.labels)} testsubset {self.testsplit} features loaded")
            # ipdb.set_trace() # 查看一下testset的label大小情况，确认背景类的id，背景id是7
            # import pdb; pdb.set_trace()
    
    def replace(self, features=None, labels=None):
        self.features = features
        self.labels = labels
        self.ntrain = len(self.labels)
        print(f"\n=== Replaced new batch of Syn Feats === \n")

    def __getitem__(self, idx):
        batch_feature = self.features[idx]
        batch_label = self.labels[idx]
        if self.classid_tolabels is not None:
            batch_label = self.classid_tolabels[batch_label]
        return batch_feature, batch_label

    def __len__(self):
        return len(self.labels)


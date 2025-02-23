import numpy as np
import torch
# import pandas as pd 
from torch.utils.data import Dataset
from util import *

import os.path
from os import path
import ipdb 

class FeaturesCls(Dataset):
     
    def __init__(self, opt, features=None, labels=None, val=False, split='seen', classes_to_train=None):
        self.root = f"{opt.dataroot}"
        self.opt = opt
        self.classes_to_train = classes_to_train
        self.classid_tolabels = None
        self.features = features
        self.labels = labels
        if self.classes_to_train is not None:
            self.classid_tolabels = {label: i for i, label in enumerate(self.classes_to_train)}
        print(self.classid_tolabels)
        # ipdb.set_trace()
        print(f"class ids for unseen classifier {self.classes_to_train}")
        if 'test' in split:
            self.loadRealFeats(syn_feature=features, syn_label=labels, split=split)

    def loadRealFeats(self, syn_feature=None, syn_label=None, split='train'):
        # 当只想训练6类的零样本分类器时，加载的确实是zero_shot_0.6_0.3_xxx.npy数据
        if 'test' in split:
            self.features = np.load(f"{self.root}/{self.opt.testsplit}_feats.npy")
            self.labels = np.load(f"{self.root}/{self.opt.testsplit}_labels.npy")
            print(f"{len(self.labels)} testsubset {self.root}/{self.opt.testsplit} features loaded")
            # ipdb.set_trace()
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

class FeaturesGAN():
    def __init__(self, opt):
        self.root = f"{opt.dataroot}"
        self.opt = opt
        # self.attribute = np.load(opt.class_embedding)
        print("loading numpy arrays")
        # self.opt.trainsplit is train_0.6_0.3
        # self.all_features.shape is [8777328, 1024], self.all_labels.shape is [8777328]
        
        self.all_features = np.load(f"{self.root}/{self.opt.trainsplit}_feats.npy", allow_pickle=True)  
        self.all_labels = np.load(f"{self.root}/{self.opt.trainsplit}_labels.npy", allow_pickle=True)
        # self.all_features = np.concatenate(self.all_features)
        # self.all_labels = np.concatenate(self.all_labels)
        
        mean_path = f"{self.root}/{self.opt.trainsplit}_mean.npy"
        print(f'loaded data from {self.opt.trainsplit}')
        K = max(self.all_labels)
        self.pos_inds = np.where(self.all_labels<K)[0] # 记录正样本的位置，list类型，长度为 2925776
        self.neg_inds = np.where(self.all_labels==K)[0] # 记录负样本的位置，list类型，长度为5851552


        unique_labels = np.unique(self.all_labels) # len(unique_labels) is 66
        self.num_bg_to_take = len(self.pos_inds)//len(unique_labels) #决定每个epoch中背景类别的训练样本多少，和其他各类保持一致

        print(f"loaded {len(self.pos_inds)} fg labels")
        print(f"loaded {len(self.neg_inds)} bg labels ")
        print(f"bg indexes for each epoch {self.num_bg_to_take}")


        self.features_mean = np.zeros((max(unique_labels) + 1 , self.all_features.shape[1]))
        # if path.exists(mean_path):
        #     self.features_mean = np.load(mean_path)
        # else:
        #     for label in unique_labels:
        #         label_inds = np.where(self.all_labels==label)[0]
        #         self.features_mean[label] = self.all_features[label_inds].mean(axis=0)
        #     np.save(mean_path, self.features_mean)
        
    def epochData(self, include_bg=False):
        fg_inds = np.random.permutation(self.pos_inds)
        inds = np.random.permutation(fg_inds)[:int(self.opt.gan_epoch_budget)]
        if include_bg:
            bg_inds = np.random.permutation(self.neg_inds)[:self.num_bg_to_take]
            inds = np.random.permutation(np.concatenate((fg_inds, bg_inds)))[:int(self.opt.gan_epoch_budget)]
        features = self.all_features[inds]
        labels = self.all_labels[inds]    
        return features, labels

    def getBGfeats(self, num=1000):
        bg_inds = np.random.permutation(self.neg_inds)[:num]
        print(f"{len(bg_inds)} ")
        return self.all_features[bg_inds], self.all_labels[bg_inds]
    def __len__(self):
        return len(self.all_labels)


class FeaturesGAN_multilabels():
    def __init__(self, opt):
        self.root = f"{opt.dataroot}"
        self.opt = opt
        # self.attribute = np.load(opt.class_embedding)
        print("loading numpy arrays")
        # self.opt.trainsplit is train_0.6_0.3
        # self.all_features.shape is [8777328, 1024], self.all_labels.shape is [8777328]
        
        self.all_features = np.load(f"{self.root}/{self.opt.trainsplit}_feats.npy", allow_pickle=True)  
        self.all_labels = np.load(f"{self.root}/{self.opt.trainsplit}_labels.npy", allow_pickle=True)
        self.all_softlabels = np.load(f"{self.root}/{self.opt.trainsplit}_softlabels.npy", allow_pickle=True)
        self.all_sampgtlabels = np.load(f"{self.root}/{self.opt.trainsplit}_sampgtlabels.npy", allow_pickle=True)
        # self.all_features = np.concatenate(self.all_features)
        # self.all_labels = np.concatenate(self.all_labels)
        ipdb.set_trace()
        mean_path = f"{self.root}/{self.opt.trainsplit}_mean.npy"
        print(f'loaded data from {self.opt.trainsplit}')
        K = max(self.all_labels)
        self.pos_inds = np.where(self.all_labels<K)[0] # 记录正样本的位置，list类型，长度为 2925776
        self.neg_inds = np.where(self.all_labels==K)[0] # 记录负样本的位置，list类型，长度为5851552


        unique_labels = np.unique(self.all_labels) # len(unique_labels) is 66
        self.num_bg_to_take = len(self.pos_inds)//len(unique_labels) #决定每个epoch中背景类别的训练样本多少，和其他各类保持一致

        print(f"loaded {len(self.pos_inds)} fg labels")
        print(f"loaded {len(self.neg_inds)} bg labels ")
        print(f"bg indexes for each epoch {self.num_bg_to_take}")


        self.features_mean = np.zeros((max(unique_labels) + 1 , self.all_features.shape[1]))
        # if path.exists(mean_path):
        #     self.features_mean = np.load(mean_path)
        # else:
        #     for label in unique_labels:
        #         label_inds = np.where(self.all_labels==label)[0]
        #         self.features_mean[label] = self.all_features[label_inds].mean(axis=0)
        #     np.save(mean_path, self.features_mean)
        
    def epochData(self, include_bg=False):
        fg_inds = np.random.permutation(self.pos_inds)
        inds = np.random.permutation(fg_inds)[:int(self.opt.gan_epoch_budget)]
        if include_bg:
            bg_inds = np.random.permutation(self.neg_inds)[:self.num_bg_to_take]
            inds = np.random.permutation(np.concatenate((fg_inds, bg_inds)))[:int(self.opt.gan_epoch_budget)]
        features = self.all_features[inds]
        labels = self.all_labels[inds]    
        return features, labels

    def getBGfeats(self, num=1000):
        bg_inds = np.random.permutation(self.neg_inds)[:num]
        print(f"{len(bg_inds)} ")
        return self.all_features[bg_inds], self.all_labels[bg_inds]
    def __len__(self):
        return len(self.all_labels)
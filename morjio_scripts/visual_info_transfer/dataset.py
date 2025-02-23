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
            if len(set(self.labels)) == 16:
                self.classid_tolabels = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 80:15}
            elif len(set(self.labels)) == 8:
                self.classid_tolabels = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 80:7}
            elif len(set(self.labels)) == 21:
                self.classid_tolabels = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:10, 11:11, 12:12, 13:13, 14:14, 15:15, 16:16, 17:17, 18:18, 19:19, 80:20}
            elif len(set(self.labels)) == 179: # 178 fg + 1 bg
                self.classid_tolabels = {12: 0, 19: 1, 20: 2, 29: 3, 50: 4, 62: 5, 68: 6, 70: 7, 92: 8, 105: 9, 116: 10, 122: 11, 125: 12, 129: 13, 130: 14, 135: 15, 141: 16, 146: 17, 154: 18, 158: 19, 160: 20, 163: 21, 166: 22, 171: 23, 195: 24, 208: 25, 213: 26, 221: 27, 222: 28, 235: 29, 237: 30, 239: 31, 250: 32, 265: 33, 281: 34, 290: 35, 293: 36, 294: 37, 306: 38, 309: 39, 315: 40, 316: 41, 322: 42, 325: 43, 330: 44, 347: 45, 348: 46, 351: 47, 356: 48, 361: 49, 363: 50, 364: 51, 365: 52, 367: 53, 380: 54, 388: 55, 397: 56, 399: 57, 404: 58, 406: 59, 409: 60, 412: 61, 415: 62, 426: 63, 427: 64, 431: 65, 434: 66, 448: 67, 455: 68, 478: 69, 479: 70, 481: 71, 485: 72, 487: 73, 505: 74, 508: 75, 512: 76, 515: 77, 531: 78, 534: 79, 537: 80, 540: 81, 542: 82, 550: 83, 556: 84, 559: 85, 560: 86, 571: 87, 579: 88, 598: 89, 601: 90, 602: 91, 609: 92, 617: 93, 618: 94, 619: 95, 624: 96, 631: 97, 633: 98, 634: 99, 637: 100, 647: 101, 650: 102, 656: 103, 662: 104, 670: 105, 677: 106, 685: 107, 687: 108, 690: 109, 692: 110, 721: 111, 732: 112, 751: 113, 753: 114, 754: 115, 757: 116, 758: 117, 763: 118, 782: 119, 783: 120, 784: 121, 786: 122, 787: 123, 795: 124, 802: 125, 804: 126, 809: 127, 811: 128, 819: 129, 821: 130, 828: 131, 830: 132, 851: 133, 858: 134, 872: 135, 885: 136, 886: 137, 890: 138, 891: 139, 907: 140, 912: 141, 913: 142, 916: 143, 919: 144, 936: 145, 937: 146, 938: 147, 940: 148, 941: 149, 943: 150, 951: 151, 973: 152, 982: 153, 984: 154, 990: 155, 991: 156, 993: 157, 1011: 158, 1015: 159, 1027: 160, 1029: 161, 1030: 162, 1046: 163, 1047: 164, 1048: 165, 1052: 166, 1056: 167, 1115: 168, 1117: 169, 1128: 170, 1143: 171, 1144: 172, 1147: 173, 1149: 174, 1158: 175, 1164: 176, 1192: 177, 1203:178}
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
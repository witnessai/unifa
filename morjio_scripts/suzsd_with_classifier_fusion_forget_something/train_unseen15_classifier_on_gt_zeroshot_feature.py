from arguments import parse_args
import random
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import os
import numpy as np
from dataset import FeaturesCls
from splits import get_asd_zero_shot_class_ids
import ipdb 
from util import *
import torch.nn as nn
import torch.optim as optim
from dataset import *
from cls_models import ClsModelTrain
from splits import get_asd_zero_shot_class_ids, get_asd_zero_shot_class_labels, get_unseen_class_ids, get_unseen_class_labels

class TrainCls():
    def __init__(self, opt):
        # opt.dataset: coco, opt.classes_split: 65_8_7, zero_shot_class_ids is [ 4, 15, 28, 29, 48, 61, 64]
        zero_shot_class_ids = get_unseen_class_ids(opt.dataset, split=opt.classes_split)
        # [len(zero_shot_class_ids)] represent bg class, 
        self.classes_to_train = np.concatenate((zero_shot_class_ids, [80]))
        self.opt = opt
        # self.classes represent classnames
        self.classes = get_unseen_class_labels(self.opt.dataset, split=opt.classes_split)
        self.best_acc = -100000
        self.isBestIter = False
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.NLLLoss()

        self.dataset = None 
        self.val_accuracies = []
        self.init_model()
        self.best_epoch = 0

    def init_model(self):
        # 只训练未知类，加Train后缀表示没有softmax
        self.classifier = ClsModelTrain(num_classes=len(self.classes_to_train))
        # ipdb.set_trace()
        self.classifier.cuda()
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.opt.lr_cls, betas=(0.5, 0.999))

    def initDataSet(self, features, labels):
        # self.classes_to_train is [ 4, 15, 28, 29, 48, 61, 64, 80]
        self.dataset = FeaturesCls(self.opt, features=features, labels=labels, split='train', classes_to_train=self.classes_to_train) 
        self.test_dataset = FeaturesCls(self.opt, split='test', classes_to_train=self.classes_to_train)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.opt.batch_size, num_workers=4, shuffle=True, pin_memory=True)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.opt.batch_size*50, num_workers=4, shuffle=True, pin_memory=True)
        
    def updateDataSet(self, features, labels):
        self.dataloader.dataset.replace(features, labels)

    def __call__(self, features=None, labels=None, gan_epoch=0):
        self.isBestIter = False
        self.gan_epoch = gan_epoch

        if self.dataset  is None:
            self.initDataSet(features, labels)
            self.valacc, self.all_acc, _ = val(self.test_dataloader, self.classifier, self.criterion, self.opt, 0, verbose="Test")
            self.val_accuracies.append(self.all_acc)
        else:
            self.updateDataSet(features, labels)
        
        self.init_model()
        self.trainEpochs()
        self.best_acc = max(self.best_acc, self.valacc)

    def trainEpochs(self):
        for epoch in range(self.opt.nepoch_cls):
            self.classifier.train()
            loss_epoch = 0
            preds_all = []
            gt_all = []
            for ite, (in_feat, in_label)  in enumerate(self.dataloader):
                in_feat = in_feat.type(torch.float).cuda()
                in_label = in_label.cuda()
                preds = self.classifier(feats=in_feat, classifier_only=True)
                loss = self.criterion(preds, in_label)

                loss_epoch+=loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                preds_all.append(preds.data.cpu().numpy())
                gt_all.append(in_label.data.cpu().numpy())
                
                if ite % 30 == 29:
                    print(f'Cls Train Epoch [{epoch+1:02}/{self.opt.nepoch_cls}] Iter [{ite:05}/{len(self.dataloader)}]{ite/len(self.dataloader) * 100:02.3f}% Loss: {loss_epoch/ite :0.4f} lr: {get_lr(self.optimizer):0.6f}')
            # validate on test set
            adjust_learning_rate(self.optimizer, epoch, self.opt)

            self.valacc, self.all_acc, c_mat_test = val(self.test_dataloader, self.classifier, self.criterion, self.opt, epoch, verbose="Test")
            self.val_accuracies.append(self.all_acc)

            if self.best_acc <= self.valacc:
                torch.save({'state_dict': self.classifier.state_dict(), 'epoch': epoch}, f"{self.opt.outname}/classifier_best.pth")
                print(f"saved best model best accuracy : {self.valacc:0.6f}")
                self.isBestIter = True
                np.save(f'{self.opt.outname}/confusion_matrix_Test.npy', c_mat_test)
            self.best_acc = max(self.best_acc, self.valacc)
            if self.isBestIter:
                self.best_epoch = self.gan_epoch
                torch.save({'state_dict': self.classifier.state_dict(), 'epoch': epoch}, f"{self.opt.outname}/classifier_best_latest.pth")
        
        _,_, c_mat_train = compute_per_class_acc(np.concatenate(gt_all), np.concatenate(preds_all), self.opt, verbose='Train')
        np.save(f'{self.opt.outname}/confusion_matrix_Train.npy', c_mat_train)
        torch.save({'state_dict': self.classifier.state_dict(), 'epoch': epoch}, f"{self.opt.outname}/classifier_latest.pth")

        print(f"[{self.best_epoch:04}] best model accuracy {self.best_acc}")




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
print(set(labels))
# ipdb.set_trace()
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



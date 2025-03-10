
from __future__ import print_function
import torch
import torch.optim as optim
from util import *
import torch.nn as nn
# from mmdetection.tools.faster_rcnn_utils import *
from torch.utils.data import DataLoader
import numpy as np
from dataset_15_5 import *
from cls_models import ClsModelTrain, ClsModelTrainContrastive
from splits import get_asd_zero_shot_class_ids, get_asd_zero_shot_class_labels, get_asd_few_zero_shot_class_ids, get_asd_few_zero_shot_class_labels, get_asd_base_few_zero_shot_class_ids, get_asd_base_few_zero_shot_class_labels
import ipdb 
from supervised_contrastive_loss import SupervisedContrastiveLoss
import time

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class TrainCls():
    def __init__(self, opt):
        # opt.dataset: coco, opt.classes_split: 65_8_7, zero_shot_class_ids is [ 4, 15, 28, 29, 48, 61, 64]
    
        if opt.traincls_classifier == 'zero_shot_set1':
            class_ids = get_asd_zero_shot_class_ids(opt.dataset, split=opt.classes_split, fs_set=1)
            class_labels = get_asd_zero_shot_class_labels(opt.dataset, split=opt.classes_split, fs_set=1)
        elif opt.traincls_classifier == 'zero_shot_set2':
            class_ids = get_asd_zero_shot_class_ids(opt.dataset, split=opt.classes_split, fs_set=2)
            class_labels = get_asd_zero_shot_class_labels(opt.dataset, split=opt.classes_split, fs_set=2)
        elif opt.traincls_classifier == 'zero_shot_set3':
            class_ids = get_asd_zero_shot_class_ids(opt.dataset, split=opt.classes_split, fs_set=3)
            class_labels = get_asd_zero_shot_class_labels(opt.dataset, split=opt.classes_split, fs_set=3)
        elif opt.traincls_classifier == 'few_zero_shot':
            class_ids = get_asd_few_zero_shot_class_ids(opt.dataset, split=opt.classes_split)
            class_labels = get_asd_few_zero_shot_class_labels(opt.dataset, split=opt.classes_split)
        elif opt.traincls_classifier == 'base_few_zero_shot':
            class_ids = get_asd_base_few_zero_shot_class_ids(opt.dataset, split=opt.classes_split)
            class_labels = get_asd_base_few_zero_shot_class_labels(opt.dataset, split=opt.classes_split)
        # [len(zero_shot_class_ids)] represent bg class, 
        self.classes_to_train = np.concatenate((class_ids, [20]))
        self.opt = opt
        # self.classes represent classnames
        self.classes = class_labels
        self.best_acc = -100000
        self.best_bg_acc_when_fg_avg_lt_50 = -100000
        self.best_bg_acc_when_fg_avg_lt_60 = -100000
        self.best_fg_acc = -100000
        self.isBestIter = False
        if opt.supconloss:
            self.criterion1 = nn.CrossEntropyLoss() 
            self.criterion2 = SupervisedContrastiveLoss()
            self.criterion = [self.criterion1, self.criterion2]
        else:
            self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.NLLLoss()

        self.dataset = None 
        self.val_accuracies = []
        self.init_model()
        self.best_epoch = 0
        self.best_bg_epoch = 0
        self.best_fg_epoch = 0
        self.best_bg_epoch_lt_60 = 0


    def init_model(self):
        # 只训练未知类，加Train后缀表示没有softmax
        if self.opt.supconloss:
            self.classifier = ClsModelTrainContrastive(num_classes=len(self.classes_to_train))
        else:
            self.classifier = ClsModelTrain(num_classes=len(self.classes_to_train))
            
        # ipdb.set_trace()
        self.classifier.cuda()
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=self.opt.lr_cls, betas=(0.5, 0.999))

    def initDataSet(self, features, labels):
        # self.classes_to_train is [ 4, 15, 28, 29, 48, 61, 64, 80]
        # ipdb.set_trace()
        self.dataset = FeaturesCls(self.opt, features=features, labels=labels, split='train', classes_to_train=self.classes_to_train) 
        self.test_dataset = FeaturesCls(self.opt, split='test', classes_to_train=self.classes_to_train)
        self.dataloader = DataLoader(dataset=self.dataset, batch_size=self.opt.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.opt.batch_size*50, num_workers=8, shuffle=True, pin_memory=True)
        
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
                
                if self.opt.supconloss:
                    preds, emb = self.classifier(feats=in_feat, classifier_only=True)
                    loss1 = self.criterion1(preds, in_label)
                    probs = torch.softmax(preds, 1)
                    loss2 = self.criterion2(probs, in_label, ious)
                    loss = loss1 + loss2
                    ious = torch.ones(in_label.shape[0]).cuda()
                elif self.opt.mixup:
                    # t0 = time.time()
                    mixed_x, y_a, y_b, lam = mixup_data(in_feat, in_label)
                    # t1 = time.time()
                    # print(t1-t0)
                    preds = self.classifier(feats=mixed_x, classifier_only=True)
                    # t2 = time.time()
                    # print(t2-t1)
                    loss = mixup_criterion(nn.CrossEntropyLoss(), preds, y_a, y_b, lam)
                    # t3 = time.time()
                    # print(t3-t2)
                    # ipdb.set_trace()
                else:
                    # t0 = time.time()
                    preds = self.classifier(feats=in_feat, classifier_only=True)
                    # t1 = time.time()
                    # print(t1-t0)
                    loss = self.criterion(preds, in_label)
                    # t2 = time.time()
                    # print(t2-t1)
                    # ipdb.set_trace()
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

            if self.all_acc[:-1].mean() > 0.5:
                if self.best_bg_acc_when_fg_avg_lt_50 <=  self.all_acc[-1]:
                    self.best_bg_acc_when_fg_avg_lt_50 = self.all_acc[-1]
                    torch.save({'state_dict': self.classifier.state_dict(), 'epoch': epoch}, f"{self.opt.outname}/classifier_best_bg.pth")
                    print(f"saved best bg model best accuracy : {self.best_bg_acc_when_fg_avg_lt_50:0.6f}")
                    self.best_bg_epoch = self.gan_epoch
            if self.all_acc[:-1].mean() > 0.6:
                if self.best_bg_acc_when_fg_avg_lt_60 <=  self.all_acc[-1]:
                    self.best_bg_acc_when_fg_avg_lt_60 = self.all_acc[-1]
                    torch.save({'state_dict': self.classifier.state_dict(), 'epoch': epoch}, f"{self.opt.outname}/classifier_best_bg_lt_60.pth")
                    print(f"saved best bg 60 model best accuracy : {self.best_bg_acc_when_fg_avg_lt_60:0.6f}")
                    self.best_bg_epoch_lt_60 = self.gan_epoch
            if self.best_fg_acc <= self.all_acc[:-1].mean():
                self.best_fg_acc = self.all_acc[:-1].mean()
                torch.save({'state_dict': self.classifier.state_dict(), 'epoch': epoch}, f"{self.opt.outname}/classifier_best_fg.pth")
                print(f"saved best fg model best accuracy : {self.best_fg_acc:0.6f}")
                self.best_fg_epoch = self.gan_epoch
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
        print(f"[{self.best_bg_epoch:04}] best bg model accuracy {self.best_bg_acc_when_fg_avg_lt_50}")
        print(f"[{self.best_fg_epoch:04}] best fg model accuracy {self.best_fg_acc}")
        print(f"[{self.best_bg_epoch_lt_60:04}] best bg 60 model accuracy {self.best_bg_acc_when_fg_avg_lt_60}")








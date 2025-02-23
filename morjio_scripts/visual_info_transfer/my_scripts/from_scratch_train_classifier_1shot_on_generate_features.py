import torch
import torch.nn as nn
import ipdb 
import os 
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
             discriminant_analysis, random_projection)
import torch.optim as optim
from torch.utils.data import dataset
from torch.utils.data import dataloader

## define network architecture of GAN
class MLP_G_text_emb(nn.Module):
    def __init__(self, pretrained_netG):
        super(MLP_G_text_emb, self).__init__()
        attSize = 512
        nz = 300
        ngh = 4096
        resSize = 1024
        self.fc1 = nn.Linear(attSize + nz, ngh)
        self.fc2 = nn.Linear(ngh, resSize)
        self.lrelu = nn.LeakyReLU(0.2, True)
        #self.prelu = nn.PReLU()
        self.relu = nn.ReLU(True)
        self.fc1.weight =  nn.Parameter(pretrained_netG['state_dict']['fc1.weight'].cpu())
        self.fc1.bias = nn.Parameter(pretrained_netG['state_dict']['fc1.bias'].cpu())
        self.fc2.weight = nn.Parameter(pretrained_netG['state_dict']['fc2.weight'].cpu())
        self.fc2.bias = nn.Parameter(pretrained_netG['state_dict']['fc2.bias'].cpu())

    def forward(self, noise, att, gen_labels):
        # uneven batch size
        repeat_num = int(noise.shape[0]/att.shape[0])
        att = att.repeat(repeat_num, 1)
        gen_labels = gen_labels.repeat(repeat_num, 1)
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h, gen_labels

def get_z_random(class_num):
    """
    returns normal initialized noise tensor 
    """
    batch_size = 250 * class_num
    nz = 300
    z = torch.FloatTensor(batch_size, nz)
    z.normal_(0, 1)
    return z

def generate_features(text_embedding, netG):
    COCO_NOVEL_CLASSES_AND_BG = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78, 80] 
    
    # COCO_ALL_CLASSES_AND_BG = list(range(81))
    COCO_CAT_IDS = COCO_NOVEL_CLASSES_AND_BG
    input_attv = text_embedding[COCO_CAT_IDS]
    noise = get_z_random(class_num=len(COCO_CAT_IDS))
    noise = noise.cuda()
    input_attv = input_attv.cuda()
    gen_labels = torch.Tensor(list(range(len(COCO_CAT_IDS)))).cuda().unsqueeze(1)
    gen_feats = netG(noise, input_attv, gen_labels)
    return gen_feats

# tmp_test, 48张图片， 420个标注
# json_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/few_shot_ann_65_15_split/rahman_fsd_full_box_1shot_trainval.json'
# import json 
# with open(json_path) as f:
#     ft_data = json.load(f)

## load text embedding
text_emb_model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_65_15/0.6725_best_acc_in_testdata_20230208_text_embedding/gen_best.pth'
text_emb_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_with_bg_switch_bg_for_mmdet2.npy'
pretrained_netG = torch.load(text_emb_model_path)
netG = MLP_G_text_emb(pretrained_netG)
netG = netG.cuda()
text_embedding = torch.from_numpy(np.load(text_emb_path))  # text_embedding.shape is (81, 512)
text_embedding = text_embedding.float()
text_embedding = text_embedding.cuda()



## load classifier from pretrained det model 
np.set_printoptions(suppress=True)
# torch.set_printoptions(precision=3,sci_mode=False)
class Model(nn.Module):
    def __init__(self, tar_size):
        super().__init__()
        self.fc = nn.Linear(tar_size[1], tar_size[0])
    def forward(self, x):
        cls_logits = self.fc(x)
        return cls_logits
tar_size = [16, 1024]
classifier_from_det = Model(tar_size)



## finetune on ground-truth and synthesized few-shot visual features
def filter_base_data_and_remap_id(feats, labels):
    gt_feats = []
    gt_labels = []
    COCO_NOVEL_CLASSES_AND_BG = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78, 80] 
    for cls_id in COCO_NOVEL_CLASSES_AND_BG:
        gt_feats.append(feats[labels==cls_id])
        gt_labels.append(labels[labels==cls_id])
    gt_feats = torch.cat(gt_feats)
    gt_labels = torch.cat(gt_labels)

    # ID_MAP = {v:i for i, v in enumerate(COCO_NOVEL_CLASSES_AND_BG)}
    for i, v in enumerate(COCO_NOVEL_CLASSES_AND_BG):
        gt_labels[gt_labels==v] = i 
    return gt_feats, gt_labels
    
gt_feats = torch.from_numpy(np.load('/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/1shot/finetuneset/few_shot_15_and_bg_train_0.6_0.3_feats.npy'))
gt_labels = torch.from_numpy(np.load('/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/1shot/finetuneset/few_shot_15_and_bg_train_0.6_0.3_labels.npy'))
gt_feats, gt_labels = filter_base_data_and_remap_id(gt_feats, gt_labels)



class Valdata(dataset.Dataset):
    def __init__(self, features, labels):
        super(Valdata, self).__init__()
        self.features = features 
        self.labels = labels

    def __getitem__(self, index):
        feat = self.features[index]
        label = self.labels[index]
        return feat, label

    def __len__(self):
        return len(self.labels)
    

def get_batch_gt_data():
    COCO_NOVEL_CLASSES_AND_BG = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78, 80] 
    COCO_ID = list(range(len(COCO_NOVEL_CLASSES_AND_BG)))
    batch_gt_feats = []
    batch_gt_labels = []
    bg_sample_num = 100
    fg_repeat_num = 10
    for idx in COCO_ID:
        if idx is not COCO_ID[-1]:
            batch_gt_feats.append(gt_feats[gt_labels==idx].repeat(fg_repeat_num, 1))
            batch_gt_labels.append(gt_labels[gt_labels==idx].unsqueeze(1).repeat(fg_repeat_num, 1).squeeze(1))
        else:
            bg_feats = gt_feats[gt_labels==idx]
            bg_labels = gt_labels[gt_labels==idx]
            import random
            index_value = random.sample(list(enumerate(bg_labels)), bg_sample_num)
            index = [x[0] for x in index_value]
            batch_gt_feats.append(bg_feats[index])
            batch_gt_labels.append(bg_labels[index])
    return batch_gt_feats, batch_gt_labels

def validation(model, test_dataloader):
    inst_total = np.zeros(tar_size[0])
    right_total = np.zeros(tar_size[0])
    wrong_total = np.zeros(tar_size[0])
    confusion_matrix = np.zeros((tar_size[0], tar_size[0]))
    confusion_matrix_base_few_incluede_bg = np.zeros((3, 3))
    for features, gt_labels in tqdm(test_dataloader):
        features = features
        gt_labels = gt_labels
        outputs = model(features)
        pred_labels = torch.argmax(outputs, 1)
        right_index = gt_labels == pred_labels
        for i in range(tar_size[0]):
            inst_total[i] += torch.sum(gt_labels[right_index]==i)
            right_total[i] += torch.sum(pred_labels[right_index]==i)
        wrong_index = gt_labels != pred_labels
        for i in range(tar_size[0]):
            inst_total[i] += torch.sum(gt_labels[wrong_index]==i)
            wrong_total[i] += torch.sum(pred_labels[wrong_index]==i)
        for i in range(tar_size[0]):
            for j in range(tar_size[0]):
                num = torch.sum((gt_labels==i) & (pred_labels==j))
                confusion_matrix[i, j] += num

    acc_total = np.zeros(tar_size[0])
    for i in range(len(inst_total)):
        acc_total[i] = 1.0*right_total[i]/inst_total[i]
    COCO_NOVEL_CLASSES_AND_BG = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78, 80] 
    COCO_ALL_CLASSES_AND_BG = list(range(81))
    COCO_BASE_CLASSES = list(set(COCO_ALL_CLASSES_AND_BG)-set(COCO_NOVEL_CLASSES_AND_BG))
    COCO_NOVEL_CLASSES = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78] 
    
    print(acc_total)
    print(np.mean(acc_total))
    print(confusion_matrix)
    # ipdb.set_trace()

features = torch.from_numpy(np.load('/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/1shot/test_0.6_0.3_feats.npy'))
labels = torch.from_numpy(np.load('/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/1shot/test_0.6_0.3_labels.npy'))
# ipdb.set_trace()
labels[labels==80] = 15
features = features.cuda()
labels = labels.cuda()
test_dataset = Valdata(features, labels)
test_dataloader = dataloader.DataLoader(dataset=test_dataset,batch_size=65536,shuffle=False,num_workers=0,drop_last=False)

epoch_num = 100
eval_interval = 5
model_save_interval = 5
train_data = 'all' 
# train_data = 'gen' 
# train_data = 'gt' 
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier_from_det.parameters(), lr=0.001, momentum=0.9)
classifier_from_det = classifier_from_det.cuda()
# validation(classifier_from_det, test_dataloader)
for epoch in tqdm(range(epoch_num)):
    batch_gen_feats, batch_gen_labels = generate_features(text_embedding, netG)
    batch_gt_feats, batch_gt_labels = get_batch_gt_data()
    batch_gt_feats = torch.cat(batch_gt_feats).cuda()
    batch_gt_labels = torch.cat(batch_gt_labels).unsqueeze(1).cuda()
    if train_data == 'all':
        batch_mix_feats = torch.cat([batch_gen_feats, batch_gt_feats])
        batch_mix_labels = torch.cat([batch_gen_labels, batch_gt_labels])
        batch_mix_labels = batch_mix_labels.long().squeeze(1)
    elif train_data == 'gen':
        batch_mix_feats = batch_gen_feats 
        batch_mix_labels = batch_gen_labels.long().squeeze(1)
    elif train_data == 'gt':
        batch_mix_feats = batch_gt_feats 
        batch_mix_labels = batch_gt_labels.long().squeeze(1)
    else:
        print('error')
        exit()

    outputs = classifier_from_det(batch_mix_feats)
    loss = criterion(outputs, batch_mix_labels)
    loss.backward()
    optimizer.step()
    if epoch % eval_interval == 0:
        validation(classifier_from_det, test_dataloader)
    if epoch % model_save_interval == 0:
        torch.save({'state_dict': classifier_from_det.state_dict(), 'epoch': epoch}, f"/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/work_dirs/fsd_65_15_r101_fpn_coco_1shot-fine-tuning/from_scratch_train_classifier_1shot_on_gen_featsepoch{epoch}.pth")
ipdb.set_trace()


## shell command
# CUDA_VISIBLE_DEVICES=0 python from_scratch_train_classifier_1shot_on_generate_features.py

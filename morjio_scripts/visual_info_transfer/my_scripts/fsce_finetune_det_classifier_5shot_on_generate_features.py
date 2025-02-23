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
    input_attv = text_embedding[COCO_NOVEL_CLASSES_AND_BG]
    noise = get_z_random(class_num=len(COCO_NOVEL_CLASSES_AND_BG))
    noise = noise.cuda()
    input_attv = input_attv.cuda()
    gen_labels = torch.Tensor(COCO_NOVEL_CLASSES_AND_BG).cuda().unsqueeze(1)
    gen_feats = netG(noise, input_attv, gen_labels)
    return gen_feats

## load text embedding
text_emb_model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_65_15/fsce/5shot/0.6507_best_acc_in_testdata_20230303_text_embedding/gen_best.pth'
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
        model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/work_dirs/fsd_65_15_fsce_r101_fpn_coco_5shot-fine-tuning/iter_30000.pth'
        checkpoint = torch.load(model_path)
        
        COCO_NOVEL_CLASSES_AND_BG = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78, 80] # len(COCO_NOVEL_CLASSES_AND_BG) is 16
        COCO_IDMAP = {v: i for i, v in enumerate(COCO_NOVEL_CLASSES_AND_BG)}
        param_name = 'roi_head.bbox_head.fc_cls'
        is_weight_list = [True] # fsce does not set bias
        tmp_weight = nn.Linear(tar_size[1], tar_size[0])
        for is_weight in is_weight_list:
            weight_name = param_name + ('.weight' if is_weight else '.bias')
            pretrained_weight = checkpoint['state_dict'][weight_name] # shape is [81, 1024] or [81]
            for idx, c in enumerate(COCO_NOVEL_CLASSES_AND_BG):
                if is_weight:
                    tmp_weight.weight[COCO_IDMAP[c]] = pretrained_weight[c]
                    # self.fc.weight[COCO_IDMAP[c]] = pretrained_weight[c].clone().detach()
                else:
                    tmp_weight.bias[COCO_IDMAP[c]] = pretrained_weight[c]
                    # self.fc.bias[COCO_IDMAP[c]] = pretrained_weight[c].clone().detach()
        weight = torch.tensor(tmp_weight.weight.clone().detach()).numpy()
        bias = torch.tensor(tmp_weight.bias.clone().detach()).numpy()
        self.fc.weight = nn.Parameter(torch.from_numpy(weight).float())
        self.fc.bias = nn.Parameter(torch.from_numpy(bias).float())

        # self.fc.weight.copy_(torch.from_numpy(weight).float())
        # self.fc.bias.copy_(torch.from_numpy(bias).float())
        # model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/work_dirs/fsd_65_15_r101_fpn_coco_10shot-fine-tuning/few_shot_classifier.pth'
        # torch.save(model_path, tmp_weight)
        # self.fc.load_state_dict(model_path)
    def forward(self, x):
        cls_logits = self.fc(x)
        return cls_logits
tar_size = [16, 1024]
classifier_from_det = Model(tar_size)



## finetune on ground-truth and synthesized few-shot visual features
gt_feats = torch.from_numpy(np.load('/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/5shot/finetuneset/train_0.6_0.3_feats.npy'))
gt_labels = torch.from_numpy(np.load('/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/5shot/finetuneset/train_0.6_0.3_labels.npy'))
gt_feats = gt_feats
gt_labels = gt_labels

bg_pointer = 0
sampling_num = 10
def get_batch_gt_data(bg_pointer):
    COCO_NOVEL_CLASSES_AND_BG = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78, 80] 
    batch_gt_feats = []
    batch_gt_labels = []
    for idx in COCO_NOVEL_CLASSES_AND_BG:
        if idx is not 80: # exclude bg
            batch_gt_feats.append(gt_feats[gt_labels==idx])
            batch_gt_labels.append(gt_labels[gt_labels==idx])
        else:
            batch_gt_feats.append(gt_feats[gt_labels==idx][bg_pointer:bg_pointer+sampling_num])
            batch_gt_labels.append(gt_labels[gt_labels==idx][bg_pointer:bg_pointer+sampling_num]) 
            bg_pointer = bg_pointer+sampling_num
    return batch_gt_feats, batch_gt_labels

def validation(model):
    features = torch.from_numpy(np.load('/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/5shot/test_0.6_0.3_feats.npy')).cuda()
    labels = torch.from_numpy(np.load('/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/5shot/test_0.6_0.3_labels.npy')).cuda()
    labels[labels==80] = 15
    train_num = labels.shape[0]
    inst_total = np.zeros(tar_size[0])
    right_total = np.zeros(tar_size[0])
    wrong_total = np.zeros(tar_size[0])
    for i in tqdm(range(train_num)):
        input = features[i]
        output = model(input)
        gt = labels[i]
        idx = torch.argmax(output)
        if gt == idx:
            inst_total[gt] += 1
            right_total[gt] += 1
        else:
            inst_total[gt] += 1
            wrong_total[idx] += 1

    acc_total = np.zeros(tar_size[0])
    for i in range(len(inst_total)):
        acc_total[i] = 1.0*right_total[i]/inst_total[i]
    print(acc_total)
    print(np.mean(acc_total))

epoch_num = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(classifier_from_det.parameters(), lr=0.001, momentum=0.9)
COCO_NOVEL_CLASSES_AND_BG = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78, 80]
COCO_IDMAP = {v: i for i, v in enumerate(COCO_NOVEL_CLASSES_AND_BG)}
classifier_from_det = classifier_from_det.cuda()
validation(classifier_from_det)
for epoch in tqdm(range(epoch_num)):
    batch_gen_feats, batch_gen_labels = generate_features(text_embedding, netG)
    batch_gt_feats, batch_gt_labels = get_batch_gt_data(bg_pointer)
    batch_gt_feats = torch.cat(batch_gt_feats).cuda()
    batch_gt_labels = torch.cat(batch_gt_labels).unsqueeze(1).cuda()
    batch_mix_feats = torch.cat([batch_gen_feats, batch_gt_feats])
    batch_mix_labels = torch.cat([batch_gen_labels, batch_gt_labels])
    batch_mix_labels = batch_mix_labels.long().squeeze(1)
    for idx, c in enumerate(COCO_NOVEL_CLASSES_AND_BG):
        batch_mix_labels[batch_mix_labels==c] = idx
    
    outputs = classifier_from_det(batch_mix_feats)
    loss = criterion(outputs, batch_mix_labels)
    loss.backward()
    optimizer.step()
    validation(classifier_from_det)
    torch.save({'state_dict': classifier_from_det.state_dict(), 'epoch': epoch}, f"/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/work_dirs/fsd_65_15_fsce_r101_fpn_coco_5shot-fine-tuning/classifier_best_finetuning_on_gen_feats_epoch{epoch}.pth")
ipdb.set_trace()

# CUDA_VISIBLE_DEVICES=2 python fsce_finetune_det_classifier_5shot_on_generate_features.py
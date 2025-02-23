import os
import ipdb 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from time import time
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
             discriminant_analysis, random_projection)
import torch.backends.cudnn as cudnn
import random

random_seed = 2
seed = random_seed
torch.manual_seed(seed) # 为CPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.	
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU，为所有GPU设置随机种子
cudnn.benchmark = True

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    batch_size = x.size()[0]
    if alpha > 0:
        lam = np.random.beta(alpha, alpha, size=batch_size)
    else:
        lam = 1

    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    ipdb.set_trace()
    lam = torch.from_numpy(lam).float().cuda().unsqueeze(1)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    if y is not None:
        y_a, y_b = y, y[index]
    else:
        y_a, y_b = None, None
    return mixed_x, y_a, y_b, lam

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

    def forward(self, noise, att):
        # uneven batch size
        noise = noise[:att.shape[0], :]
        att = att.repeat(100, 1)
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

class MLP_G_word_emb(nn.Module):
    def __init__(self, pretrained_netG):
        super(MLP_G_word_emb, self).__init__()
        attSize = 300
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

    def forward(self, noise, att):
        # uneven batch size
        noise = noise[:att.shape[0], :]
        att = att.repeat(100, 1)
        h = torch.cat((noise, att), 1)
        h = self.lrelu(self.fc1(h))
        h = self.relu(self.fc2(h))
        return h

def get_z_random():
    """
    returns normal initialized noise tensor 
    """
    batch_size = 100
    nz = 300
    z = torch.FloatTensor(batch_size, nz)
    z.normal_(0, 1)
    return z

## generate feats based on text embedding
text_emb_model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_65_15/0.6725_best_acc_in_testdata_20230208_text_embedding/gen_best.pth'
text_emb_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_with_bg_switch_bg_for_mmdet2.npy'
pretrained_netG = torch.load(text_emb_model_path)
netG = MLP_G_text_emb(pretrained_netG)
netG = netG.cuda()
text_embedding = torch.from_numpy(np.load(text_emb_path))  # text_embedding.shape is (81, 512)
text_embedding = text_embedding.float()
unseen_and_bg_index = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78, 80]
iter_num = len(unseen_and_bg_index)
gen_feats = []
for i in range(iter_num):
    emb_index = unseen_and_bg_index[i]
    input_attv = text_embedding[emb_index]
    noise = get_z_random()
    noise = noise.cuda()
    input_attv = input_attv.cuda()
    gen_feats.append(netG(noise, input_attv))
gen_feats = torch.cat(gen_feats)
visual_feats_from_text_emb = gen_feats


## generate feats based on word embedding
word_emb_model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_65_15/0.5639_best_acc_in_testdata_20230208_word_embedding_fasttext/gen_best.pth'
word_emb_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/fasttext_switch_bg.npy'
pretrained_netG = torch.load(word_emb_model_path)
netG = MLP_G_word_emb(pretrained_netG)
netG = netG.cuda()
word_embedding = torch.from_numpy(np.load(word_emb_path))  # word_embedding.shape is (81, 300)
word_embedding = word_embedding.float()
unseen_and_bg_index = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78, 80]
iter_num = len(unseen_and_bg_index)
gen_feats = []
for i in range(iter_num):
    emb_index = unseen_and_bg_index[i]
    input_attv = word_embedding[emb_index]
    noise = get_z_random()
    noise = noise.cuda()
    input_attv = input_attv.cuda()
    gen_feats.append(netG(noise, input_attv))
gen_feats = torch.cat(gen_feats)
visual_feats_from_word_emb = gen_feats


## sample gt feats from test_0.6_0.3_feats.npy and test_0.6_0.3_labels.npy
gt_feats = torch.from_numpy(np.load('/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/tfa/10shot/test_0.6_0.3_feats.npy'))
gt_labels = torch.from_numpy(np.load('/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/tfa/10shot/test_0.6_0.3_labels.npy'))
gt_feats = gt_feats.float()
gt_feats = gt_feats.cuda()
gt_labels = gt_labels.cuda()
label_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 80]
visual_feats_from_gt_test = []
for item in label_set:
    visual_feats_from_gt_test.append(gt_feats[gt_labels==item][:100])
visual_feats_from_gt_test = torch.cat(visual_feats_from_gt_test)


## t-SNE visualize
def plot_embedding_v1(X, Y, title=None):
    target_name = ['airplane', 'train', 'parking meter', 'cat', 'bear', 'suitcase', 'frisbee', 'snowboard', 'fork', 'sandwich', 'hot dog', 'toilet', 'mouse', 'toaster', 'hair drier', 'background']
    color_set = []
    for i in range(16):
        color = plt.cm.tab20(i)
        color_set.extend([color]*100)

    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)     
    plt.figure()
    ax = plt.subplot(111)
    ax_legend_list = []
    for i in range(X.shape[0]):
        xx = plt.scatter(X[i, 0], X[i, 1], 
                 color=plt.cm.tab20(Y[i] / 16.),
                 marker='.')
        if i%100 == 0:
            ax_legend_list.append(xx)
    # plt.scatter(X[:, 0], X[:, 1], color=color_set, marker='.')
    plt.xticks([]), plt.yticks([])
    # plt.legend(target_name)
    ax.legend(ax_legend_list, target_name, bbox_to_anchor=(1.05, 1))
    if title is not None:
        plt.title(title)
    plt.subplots_adjust(right=0.7)

def plot_embedding_v2(X, Y, title=None, is_wordemb=False):
    target_name = ['airplane', 'train', 'parking meter', 'cat', 'bear', 'suitcase', 'frisbee', 'snowboard', 'fork', 'sandwich', 'hot dog', 'toilet', 'mouse', 'toaster', 'hair drier', 'background']
    color_set = []
    for i in range(16):
        color = plt.cm.tab20(i)
        color_set.extend([color]*100)
    color_text = []
    for i in range(16):
        color = plt.cm.tab20(i)
        color = np.array(list(color))
        color[color<1] = color[color<1]+0.05
        color = tuple(list(color))
        color_text.extend(color)
    
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)     
    plt.figure()
    ax = plt.subplot(111)
    ax_legend_list = []
    for i in range(X.shape[0]):
        xx = plt.scatter(X[i, 0], X[i, 1], 
                 color=plt.cm.tab20(Y[i] / 16.),
                 marker='.')
        # if i%100 == 0:
        #     if is_wordemb:
        #         plt.text(X[i, 0]-0.02, X[i, 1]-0.09, target_name[i//100], 
        #          color=plt.cm.tab20(Y[i] / 16.), 
        #          fontdict={'weight': 'bold', 'size': 6}, 
        #         #  bbox={'facecolor':'white'}
        #          )
        #         print('y')
        #     else:
        #         plt.text(X[i, 0]-0.03, X[i, 1]-0.07, target_name[i//100], 
        #          color=plt.cm.tab20(Y[i] / 16.), 
        #          fontdict={'weight': 'bold', 'size': 6}, 
        #         #  bbox={'facecolor':'white'}
        #          )
        #         print('n')
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


def plot_embedding_v3(X, Y, title=None, is_wordemb=False):
    target_name = ['airplane', 'train', 'parking meter', 'cat', 'bear', 'suitcase', 'frisbee', 'snowboard', 'fork', 'sandwich', 'hot dog', 'toilet', 'mouse', 'toaster', 'hair drier', 'background']
    color_set = []
    for i in range(16):
        color = plt.cm.tab20(i)
        color_set.extend([color]*100)
    color_text = []
    for i in range(16):
        color = plt.cm.tab20(i)
        color = np.array(list(color))
        color[color<1] = color[color<1]+0.05
        color = tuple(list(color))
        color_text.extend(color)
    
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)     
    plt.figure()
    ax = plt.subplot(111)
    ax_legend_list = []
    for i in range(X.shape[0]):
        xx = plt.scatter(X[i, 0], X[i, 1], 
                 color='gray',
                 marker='.')
        # if i%100 == 0:
        #     if is_wordemb:
        #         plt.text(X[i, 0]-0.02, X[i, 1]-0.09, target_name[i//100], 
        #          color=plt.cm.tab20(Y[i] / 16.), 
        #          fontdict={'weight': 'bold', 'size': 6}, 
        #         #  bbox={'facecolor':'white'}
        #          )
        #         print('y')
        #     else:
        #         plt.text(X[i, 0]-0.03, X[i, 1]-0.07, target_name[i//100], 
        #          color=plt.cm.tab20(Y[i] / 16.), 
        #          fontdict={'weight': 'bold', 'size': 6}, 
        #         #  bbox={'facecolor':'white'}
        #          )
        #         print('n')
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


tsne = manifold.TSNE(n_components=2, init='pca', random_state=random_seed)
Y = list(range(16))
Y = [x for item in Y for x in [item]*100]

# 关于颜色的定义: https://matplotlib.org/stable/tutorials/colors/colormaps.html

X_tsne = tsne.fit_transform(visual_feats_from_text_emb.detach().cpu().numpy())
# ipdb.set_trace()
# plot_embedding_v1(X_tsne, Y, "visual features generated from text embedding")
# plt.savefig("vis_feat_from_text_seed%d.png" % random_seed)
# plot_embedding_v2(X_tsne, Y, "visual features generated from text embedding")
plot_embedding_v2(X_tsne, Y, "visual features generated from text embedding")
plt.savefig("vis_feat_from_text_v2_seed%d_0817.png" % random_seed)
# plt.show()
X_tsne = tsne.fit_transform(visual_feats_from_word_emb.detach().cpu().numpy())
# plot_embedding_v1(X_tsne, Y, "visual features generated from word embedding")
# plt.savefig("vis_feat_from_word_seed%d.png" % random_seed)
# plot_embedding_v2(X_tsne, Y, "visual features generated from word embedding")
# ipdb.set_trace()
plot_embedding_v2(X_tsne, Y, is_wordemb=True)
plt.savefig("vis_feat_from_word_v2_seed%d_0817.png" % random_seed)
# plt.show()
X_tsne = tsne.fit_transform(visual_feats_from_gt_test.detach().cpu().numpy())
plot_embedding_v1(X_tsne, Y, "visual features of GT")
plt.savefig("vis_feat_from_gt_seed%d_0817.png" % random_seed)
# plot_embedding_v2(X_tsne, Y, "visual features generated from word embedding")
# plt.savefig("vis_feat_from_gt_v2_seed%d.png" % random_seed)
# plt.show()

enhanced_visual_feats_from_text_emb, _, _, _ = mixup_data(visual_feats_from_text_emb, None, alpha=1.0, use_cuda=True)
X_tsne = tsne.fit_transform(enhanced_visual_feats_from_text_emb.detach().cpu().numpy())
plot_embedding_v3(X_tsne, Y, "synthesized feature after enchancing")
plt.savefig("vis_feat_enhanced_v2_seed%d_0817.png" % random_seed)
ipdb.set_trace()

## shell command: 
# CUDA_VISIBLE_DEVICES=4 python use_trained_gan_to_gen_feats_and_cmp_with_gt_and_tsne_visualize.py
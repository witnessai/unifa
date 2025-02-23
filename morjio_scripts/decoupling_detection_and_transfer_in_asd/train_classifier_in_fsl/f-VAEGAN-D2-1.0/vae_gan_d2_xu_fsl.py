from __future__ import print_function
import argparse
import os
import random
from socket import IP_DROP_MEMBERSHIP
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
import math
import util_fewshot 
import classifier_fsl
import sys
import model
import ipdb 
import torchvision.transforms as transforms
from PIL import Image
import numpy as np 
from tqdm import tqdm
import datetime
from torchvision import models
from torch.utils.data import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='COCO', help='FLO')
parser.add_argument('--dataroot', default='data/', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gfsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--image_att10', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--image_att', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=300, help='size of the latent z vector')
parser.add_argument('--latent_size', type=int, default=300, help='size of the latent z vector')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--nepoch_classifier', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')
parser.add_argument('--gan_weight', type=float, default=100000, help='weight of the classification loss')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netD2', default='', help="path to netD (to continue training)")
parser.add_argument('--Encoder', default='', help="path to netD (to continue training)")
parser.add_argument('--netG_name', default='')
parser.add_argument('--netD_name', default='')
parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
parser.add_argument('--outname', help='folder to output data and model checkpoints')
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--save_after', type=int, default=200)
parser.add_argument('--print_every', type=int, default=1)
parser.add_argument('--val_every', type=int, default=10)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=80, help='number of all classes')
parser.add_argument('--encoder_layer_sizes', type=list, default=[2048,1024], help='number of all classes')
parser.add_argument('--decoder_layer_sizes', type=list, default=[1024, 2048], help='number of all classes')
parser.add_argument('--ud_weight', type=float, default=1, help='weight of the classification loss')
parser.add_argument('--vae_weight', type=float, default=1, help='weight of the classification loss')
parser.add_argument('--kshot', type=int, default=10, help='number of all classes')
parser.add_argument('--splitid', default='1', help='folder to output data and model checkpoints')
parser.add_argument('--novel_weight', type=float, default=1, help='size of the latent z vector')

opt = parser.parse_args()
opt.nz = opt.latent_size
print(opt)

logger = util_fewshot.Logger(opt.outname)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

class COCO_bbox_crop(Dataset):
    def __init__(self, data_root='/data1/niehui/decoupling_detection_and_transfer_in_asd/coco_bbox_crop_for_fsl/65_15_split', img_dir='merge_base_train_finetune_subset'):
        self.data_root = data_root
        self.img_dir = img_dir
        self.img_root_path = os.path.join(data_root, img_dir)
        self.img_names = os.listdir(self.img_root_path)
        self.img_names.sort()
        self.name2label = dict()
        
        print("%s dataset has %d images!" % (img_dir, len(self.img_names)))
    
    def transform(self, image):
        imageTransform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image = imageTransform(image)
        return image

    def __getitem__(self, idx):
        while True:
            img_name = self.img_names[idx]
            img_path = os.path.join(self.data_root, os.path.join(self.img_dir, img_name))
            img = Image.open(img_path).convert('RGB')
            w, h = img.size
            if min(w, h) > 5:
                break
            else:
                idx = np.random.randint(0, len(self.img_names))
        
        img = self.transform(img)
        label = np.array(int(img_name.split('.')[0].split('_')[-1])-1) 
        
        label = torch.from_numpy(label)
        self.name2label[img_name] = label

        # pre-process
        return img, label, img_name
    
    def __len__(self):
        return len(self.img_names)






if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
logger.write('Random Seed=%d\n' % (opt.manualSeed))
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util_fewshot.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

# initialize generator and discriminator
# layer_sizes, latent_size, attSize: ([1024, 2048], 312, 312)
netG = model.Decoder(opt.decoder_layer_sizes, opt.latent_size, opt.attSize) 
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

# opt.resSize=2048, opt.attSize=312, opt.ndh=2048
netD = model.MLP_CRITIC(opt)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

# opt.resSize=2048, opt.ndh=2048
netD2 = model.MLP_CRITIC_V(opt)
if opt.netD2 != '':
    netD2.load_state_dict(torch.load(opt.netD2))
print(netD2)

# layer_sizes, latent_size, attSize: ([2360, 1024], 312, 312)
Encoder = model.Encoder(opt.encoder_layer_sizes, opt.latent_size, opt.attSize)
if opt.Encoder != '':
    Encoder.load_state_dict(torch.load(opt.Encoder))
print(Encoder)


resnet = models.resnet101(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  
# Decoder(
#   (MLP): Sequential(
#     (L0): Linear(in_features=624, out_features=1024, bias=True)
#     (A0): ReLU()
#     (L1): Linear(in_features=1024, out_features=2048, bias=True)
#     (sigmoid): Sigmoid()
#   )
# )
# MLP_CRITIC(
#   (fc1): Linear(in_features=2360, out_features=2048, bias=True)
#   (fc2): Linear(in_features=2048, out_features=1, bias=True)
#   (lrelu): LeakyReLU(negative_slope=0.2, inplace)
# )
# MLP_CRITIC_V(
#   (fc1): Linear(in_features=2048, out_features=2048, bias=True)
#   (fc2): Linear(in_features=2048, out_features=1, bias=True)
#   (lrelu): LeakyReLU(negative_slope=0.2, inplace)
# )
# Encoder(
#   (MLP): Sequential(
#     (L0): Linear(in_features=2360, out_features=1024, bias=True)
#     (A0): ReLU()
#   )
#   (linear_means): Linear(in_features=1024, out_features=312, bias=True)
#   (linear_log_var): Linear(in_features=1024, out_features=312, bias=True)
# )


batch_size = opt.batch_size
data_root = '/data1/niehui/decoupling_detection_and_transfer_in_asd/coco_bbox_crop_for_fsl/65_15_split'
trainset = COCO_bbox_crop(data_root, img_dir='merge_base_train_finetune_subset')
trainsetloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)
testset = COCO_bbox_crop(data_root, img_dir='gfsl_test')
testsetloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=16)
class_attributes = np.loadtxt('word_w2v.txt', dtype='float32', delimiter=',')
class_attributes = torch.from_numpy(class_attributes.T)
attribute_base = class_attributes[:65]
attribute_novel = class_attributes[65:]


# batch_size, resSize, attSize, nz: 512, 2048, 312, 312
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
one = torch.tensor(1, dtype=torch.float)
# one = torch.FloatTensor([1])
mone = one * -1
input_res_unpair = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att_unpair = torch.FloatTensor(opt.batch_size, opt.attSize)

if opt.cuda:
    netD.cuda()
    netD2.cuda()
    netG.cuda()
    Encoder.cuda()
    resnet.cuda()
    noise = noise.cuda()
    input_res = input_res.cuda()
    input_att = input_att.cuda()
    input_res_unpair = input_res_unpair.cuda()
    input_att_unpair = input_att_unpair.cuda()
    one = one.cuda()
    mone = mone.cuda()

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, size_average=False)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
    return BCE + KLD


def sample():
    batch_feature, batch_label, batch_att = data.next_batch_uniform_class(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    batch_feature, batch_label, batch_att = data.next_batch_unpair_test(opt.batch_size)
    input_res_unpair.copy_(batch_feature)
    input_att_unpair.copy_(batch_att)


def generate_syn_feature(vae, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
        
    with torch.no_grad():
        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            syn_att.copy_(iclass_att.repeat(num, 1))
            syn_noise.normal_(0, 1)
            output = netG(syn_noise, syn_att)
            syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerE = optim.Adam(Encoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD2 = optim.Adam(netD2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerBackbone = optim.Adam(resnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    #print real_data.size()
    alpha = torch.rand(real_data.shape[0], 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates, input_att)

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

def calc_gradient_penalty2(netD, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()
    
    interpolates.requires_grad_(True)
    disc_interpolates = netD(interpolates)

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

# train a classifier on seen classes, obtain \theta of Equation (4)

best_acc = 0
for epoch in range(opt.nepoch):
    FP = 0 
    mean_lossD = 0
    mean_lossG = 0
    seen_feat_counter = {x:0 for x in list(range(65))}
    train_feature = []
    train_label = []
    # for i in range(0, data.ntrain, opt.batch_size):
    for i, data_input in enumerate(tqdm(trainsetloader)):
        ############################
        # (1) Update D network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG update
        for p in netD2.parameters(): # reset requires_grad
            p.requires_grad = True # they are set to False below in netG_v update

        for iter_d in range(opt.critic_iter):
            # sample a mini-batch
            # sample()
            netD.zero_grad()
            netD2.zero_grad()
            resnet.zero_grad()
            # train with realG
            input_img, input_label, imgnames = data_input
            batch_size = len(input_label)
            input_att = class_attributes[input_label]
            input_img, input_label = input_img.cuda(), input_label.cuda()
            input_att = input_att.cuda()
            input_res = resnet(input_img)

            input_res = input_res.reshape(-1, opt.resSize)
            


            criticD_real = netD(input_res, input_att)
            criticD_real = criticD_real.mean()
            criticD_real.backward(mone)

            # train with fakeG
            noise = torch.FloatTensor(batch_size, opt.nz)
            noise.normal_(0, 1)
            noise = noise.cuda()
            fake = netG(noise,input_att)
            criticD_fake = netD(fake.detach(), input_att)
            criticD_fake = criticD_fake.mean()
            criticD_fake.backward(one)
            
            # gradient penalty
            # gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
            # gradient_penalty.backward()

            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real #+ gradient_penalty
            optimizerD.step()

        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters(): # reset requires_grad
            p.requires_grad = False # avoid computation
        for p in netD2.parameters(): # reset requires_grad
            p.requires_grad = False # they are set to False below in netG_v update

        netG.zero_grad()
        Encoder.zero_grad()
        resnet.zero_grad()
        # netG latent code vae loss
        mean, log_var = Encoder(input_res.detach(), input_att)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn([batch_size, opt.latent_size]).cuda()
        z = eps * std + mean
        recon_x = netG(z, input_att)
        vae_loss = loss_fn(recon_x, input_res.detach(), mean, log_var)
        # netG latent code gan loss
        criticG_fake = netD(recon_x, input_att)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake
        # net G fake data
        fake_v = netG(noise, input_att)
        criticG_fake2 = netD(fake_v, input_att)
        criticG_fake2 = criticG_fake2.mean()
        G_cost += -criticG_fake2

        loss = opt.gan_weight * G_cost + opt.vae_weight * vae_loss
        loss.backward()
        optimizerG.step()
        optimizerE.step()

        max_num = 1000
        for seen_id in seen_feat_counter:
            if seen_feat_counter[seen_id] < max_num:
                left = max_num - seen_feat_counter[seen_id]
                new_res = input_res[input_label==seen_id][:left]
                new_label = input_label[input_label==seen_id][:left]
                new_num = len(new_label)
                if new_num < 1: continue
                train_feature.append(new_res.detach())
                train_label.append(new_label.detach())
                seen_feat_counter[seen_id] += new_num
        sparse_real = opt.resSize - input_res[1].gt(0).sum()

    print('[%d/%d] Wasserstein_dist: %.4f, vae_loss:%.4f'
              % (epoch, opt.nepoch, Wasserstein_D.data.item(), vae_loss.data.item()))
    logger.write('[%d/%d] Wasserstein_dist: %.4f, vae_loss:%.4f\n'
              % (epoch, opt.nepoch, Wasserstein_D.data.item(),vae_loss.data.item()))

    # evaluate the model, set G to evaluation mode
    netG.eval()
    # Generalized few-shot learning
    if opt.gfsl:
        novelclasses = torch.tensor(list(range(65, 80)))
        train_feature = torch.cat(train_feature)
        train_label = torch.cat(train_label)
        syn_feature, syn_label = generate_syn_feature(netG, novelclasses, class_attributes, opt.syn_num)
        syn_feature, syn_label = syn_feature.cuda(), syn_label.cuda()
        train_X = torch.cat((train_feature, syn_feature), 0)
        train_Y = torch.cat((train_label, syn_label), 0)
        nclass = opt.nclass_all
        cls = classifier_fsl.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, opt.nepoch_classifier, opt.syn_num, True)
        print('acc_all=%.4f, acc_base=%.4f, acc_novel=%.4f' % (cls.acc_all, cls.acc_base, cls.acc_novel))
        logger.write('acc_all=%.4f, acc_base=%.4f, acc_novel=%.4f\n' % (cls.acc_all, cls.acc_base, cls.acc_novel))
        acc = cls.acc_all
    # Few-shot learning
    else:
        syn_feature, syn_label = generate_syn_feature(netG, data.novelclasses, data.attribute, opt.syn_num) 
        cls = classifier_fsl.CLASSIFIER(syn_feature, util_fewshot.map_label(syn_label, data.novelclasses), data, data.novelclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, opt.nepoch_classifier, opt.syn_num, False)
        acc = cls.acc
        print('novel class accuracy= ', acc)
        logger.write('novel class accuracy= %.4f\n' % acc)

    # reset G to training mode
    netG.train()

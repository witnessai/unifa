from __future__ import print_function
import argparse
import sys
sys.path.append("..")
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import util
import classifier_embed_contras
import model
import losses
import torch.nn.functional as F
import ipdb 
import datetime
from torchvision import models
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np 
from tqdm import tqdm


# dt = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='coco', help='FLO')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='sent',help='att or sent')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--gzsl', type=bool, default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', type=bool, default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=2048, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=1024 , help='size of semantic features')
parser.add_argument('--nz', type=int, default=1024, help='noise for generation')
parser.add_argument('--embedSize', type=int, default=2048, help='size of embedding h')
parser.add_argument('--outzSize', type=int, default=512, help='size of non-liner projection z')

## network architechure
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator G')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator D')
parser.add_argument('--nhF', type=int, default=2048, help='size of the hidden units comparator network F')

parser.add_argument('--ins_weight', type=float, default=0.001, help='weight of the classification loss when learning G')
parser.add_argument('--cls_weight', type=float, default=0.001, help='weight of the score function when learning G')
parser.add_argument('--ins_temp', type=float, default=0.1, help='temperature in instance-level supervision')
parser.add_argument('--cls_temp', type=float, default=0.1, help='temperature in class-level supervision')

parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to training')
parser.add_argument('--lr_decay_epoch', type=int, default=100, help='conduct learning rate decay after every 100 epochs')
parser.add_argument('--lr_dec_rate', type=float, default=0.99, help='learning rate decay rate')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--manualSeed', type=int, default=3483, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--nclass_seen', type=int, default=150, help='number of all classes')

parser.add_argument('--gpus', default='4', help='the number of the GPU to use')
opt = parser.parse_args()
print(opt)

# seen_labelid = [ 0,  1,  2,  3,  5,  7,  8,  9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 79]
# unseen_labelid = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78]
# seen_unseen_labelid = seen_labelid + unseen_labelid
# map_labelid = {seen_unseen_labelid[i]:i for i in range(len(seen_unseen_labelid))}

class COCO_bbox_crop(Dataset):
    def __init__(self, data_root='/data1/niehui/decoupling_detection_and_transfer_in_asd/coco_bbox_crop_for_zsl/65_15_split', img_dir='train'):
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


os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
# data = util.DATA_LOADER(opt)
# print("# of training samples: ", data.ntrain)
data = []
netG = model.MLP_G(opt)
netMap = model.Embedding_Net(opt)
netD = model.MLP_CRITIC(opt)
F_ha = model.Dis_Embed_Att(opt)
resnet = models.resnet101(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])   
model_path = './models/' + opt.dataset
if not os.path.exists(model_path):
    os.makedirs(model_path)
if len(opt.gpus.split(','))>1:
    netG=nn.DataParallel(netG)
    netD = nn.DataParallel(netD)
    netMap = nn.DataParallel(netMap)
    F_ha = nn.DataParallel(F_ha)
    resnet = nn.DataParallel(resnet)

batch_size = opt.batch_size
data_root = '/data1/niehui/decoupling_detection_and_transfer_in_asd/coco_bbox_crop_for_zsl/65_15_split'
trainset = COCO_bbox_crop(data_root, img_dir='train')
trainsetloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)
testset = COCO_bbox_crop(data_root, img_dir='gzsl_test')
testsetloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=16)
class_attributes = np.loadtxt('word_w2v.txt', dtype='float32', delimiter=',')
class_attributes = torch.from_numpy(class_attributes.T)



attribute_seen = class_attributes[:65]
attribute_unseen = class_attributes[65:]


if opt.cuda:
    netG.cuda()
    netD.cuda()
    netMap.cuda()
    F_ha.cuda()
    resnet.cuda()

def generate_syn_feature(netG, classes, attribute, num):
    nclass = len(classes)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = netG(syn_noise, syn_att)
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label


# setup optimizer
contras_criterion = losses.SupConLoss_clear(opt.ins_temp)
import itertools
optimizerD = optim.Adam(itertools.chain(netD.parameters(), netMap.parameters(), F_ha.parameters()), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerBackbone = optim.Adam(resnet.parameters(), lr=0.0001)

def calc_gradient_penalty(netD, real_data, fake_data, input_att, batch_size):
    # print real_data.size()
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, input_att)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

# use the for-loop to save the GPU-memory
def class_scores_for_loop(embed, input_label, relation_net):
    all_scores=torch.FloatTensor(embed.shape[0],opt.nclass_seen).cuda() # all_scores.shape [2048, 150]
    for i, i_embed in enumerate(embed): # embed.shape [2048, 2048]
        expand_embed = i_embed.repeat(opt.nclass_seen, 1)#.reshape(embed.shape[0] * opt.nclass_seen, -1)
        # expand_embed.shape is [150, 2048], data.attribute_seen.shape is [150, 1024]
        all_scores[i]=(torch.div(relation_net(torch.cat((expand_embed, attribute_seen.cuda()), dim=1)),opt.cls_temp).squeeze()) # all_scores[i].shape is [150]
    score_max, _ = torch.max(all_scores, dim=1, keepdim=True) # score_max.shape is [2048, 1]
    # normalize the scores for stable training
    scores_norm = all_scores - score_max.detach() # scores_norm.shape is [2048, 150]
    mask = F.one_hot(input_label, num_classes=opt.nclass_seen).float().cuda() # mask.shape is [2048, 150]
    exp_scores = torch.exp(scores_norm) # exp_scores.shape is [2048, 150]
    log_scores = scores_norm - torch.log(exp_scores.sum(1, keepdim=True))
    cls_loss = -((mask * log_scores).sum(1) / mask.sum(1)).mean()
    return cls_loss

# It is much faster to use the matrix, but it cost much GPU memory.
def class_scores_in_matrix(embed, input_label, relation_net):
    expand_embed = embed.unsqueeze(dim=1).repeat(1, opt.nclass_seen, 1).reshape(embed.shape[0] * opt.nclass_seen, -1)
    expand_att = attribute_seen.unsqueeze(dim=0).repeat(embed.shape[0], 1, 1).reshape(
        embed.shape[0] * opt.nclass_seen, -1).cuda()
    all_scores = torch.div(relation_net(torch.cat((expand_embed, expand_att), dim=1)),opt.cls_temp).reshape(embed.shape[0],
                                                                                                    opt.nclass_seen)
    score_max, _ = torch.max(all_scores, dim=1, keepdim=True)
    scores_norm = all_scores - score_max.detach()
    mask = F.one_hot(input_label, num_classes=opt.nclass_seen).float().cuda()
    exp_scores = torch.exp(scores_norm)
    log_scores = scores_norm - torch.log(exp_scores.sum(1, keepdim=True))
    cls_loss = -((mask * log_scores).sum(1) / mask.sum(1)).mean()
    return cls_loss
        


best_gzsl_hm_acc = 0
best_zsl_acc = 0
for epoch in range(opt.nepoch): # opt.nepoch is 2000
    FP = 0
    seen_feat_counter = {x:0 for x in list(range(65))}
    train_feature = []
    train_label = []
    # for i in range(0, data.ntrain, opt.batch_size):
    for i, data_input in enumerate(tqdm(trainsetloader)):
        ############################
        # (1) Update D network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in netMap.parameters():  # reset requires_grad
            p.requires_grad = True
        for p in F_ha.parameters():  # reset requires_grad
            p.requires_grad = True

        for iter_d in range(opt.critic_iter):
            input_img, input_label, imgnames = data_input
            batch_size = len(input_label)
            input_att = class_attributes[input_label]
            input_img, input_label = input_img.cuda(), input_label.cuda()
            input_att = input_att.cuda()
            netD.zero_grad()
            netMap.zero_grad()
            resnet.zero_grad()
            
            #
            # train with realG
            # sample a mini-batch
            input_res = resnet(input_img)
            input_res = input_res.reshape(-1, opt.resSize)
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
            embed_real, outz_real = netMap(input_res)
            criticD_real = netD(input_res, input_att)
            criticD_real = criticD_real.mean()

            # CONTRASITVE LOSS
            real_ins_contras_loss = contras_criterion(outz_real, input_label)

            # train with fakeG
            noise_gen = torch.FloatTensor(batch_size, opt.nz)
            noise_gen.normal_(0, 1)
            noise_gen = noise_gen.cuda()
            fake = netG(noise_gen, input_att)
            fake_norm = fake.data[0].norm()
            sparse_fake = fake.data[0].eq(0).sum()
            criticD_fake = netD(fake.detach(), input_att)
            criticD_fake = criticD_fake.mean()

            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att, batch_size)
            Wasserstein_D = criticD_real - criticD_fake

            cls_loss_real = class_scores_for_loop(embed_real, input_label, F_ha)
            D_cost = criticD_fake - criticD_real + gradient_penalty + real_ins_contras_loss + cls_loss_real

            D_cost.backward()
            optimizerD.step()
        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = False  # avoid computation
        for p in netMap.parameters():  # reset requires_grad
            p.requires_grad = False
        for p in F_ha.parameters():  # reset requires_grad
            p.requires_grad = False

        netG.zero_grad()
        resnet.zero_grad()
        noise_gen.normal_(0, 1)
        fake = netG(noise_gen, input_att)

        embed_fake, outz_fake = netMap(fake)

        criticG_fake = netD(fake, input_att)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake

        embed_real, outz_real = netMap(input_res)

        all_outz = torch.cat((outz_fake, outz_real.detach()), dim=0)

        fake_ins_contras_loss = contras_criterion(all_outz, torch.cat((input_label, input_label), dim=0))

        cls_loss_fake = class_scores_for_loop(embed_fake, input_label, F_ha)
        errG = G_cost + opt.ins_weight * fake_ins_contras_loss + opt.cls_weight * cls_loss_fake  # + opt.ins_weight * c_errG

        errG.backward()
        optimizerG.step()

    F_ha.zero_grad()
    if (epoch + 1) % opt.lr_decay_epoch == 0:
        for param_group in optimizerD.param_groups:
            param_group['lr'] = param_group['lr'] * opt.lr_dec_rate
        for param_group in optimizerG.param_groups:
            param_group['lr'] = param_group['lr'] * opt.lr_dec_rate

    
    
    print(
        '[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, real_ins_contras_loss:%.4f, fake_ins_contras_loss:%.4f, cls_loss_real: %.4f, cls_loss_fake: %.4f'
        % (epoch, opt.nepoch, D_cost, G_cost, Wasserstein_D, real_ins_contras_loss, fake_ins_contras_loss, cls_loss_real, cls_loss_fake))

    # evaluate the model, set G to evaluation mode
    netG.eval()

    for p in netMap.parameters():  # reset requires_grad
        p.requires_grad = False

    unseenclasses = list(range(65, 80))
    if opt.gzsl: # Generalized zero-shot learning
        train_feature = torch.cat(train_feature)
        train_label = torch.cat(train_label)
       
        # train_feature = train_feature.cuda()
        # train_label = train_label.cuda()
        syn_feature, syn_label = generate_syn_feature(netG, unseenclasses, class_attributes, opt.syn_num)
        syn_feature = syn_feature.cuda()
        syn_label = syn_label.cuda()
        train_X = torch.cat((train_feature, syn_feature), 0)
        train_Y = torch.cat((train_label, syn_label), 0)

        nclass = opt.nclass_all
        cls = classifier_embed_contras.CLASSIFIER(train_X, train_Y, netMap, opt.embedSize, testsetloader, nclass, opt.cuda,
                                                  opt.classifier_lr, 0.5, 25, opt.batch_size,
                                                  True, resnet)
        print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
        if cls.H > best_gzsl_hm_acc:
            best_gzsl_hm_acc = cls.H
            path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/decoupling_detection_and_transfer_in_asd'
            torch.save({'state_dict': cls.model.state_dict(), 'epoch': epoch}, f'{path}/gzsl_best_{cls.H}.pth')

    else:  # conventional zero-shot learning
        syn_feature, syn_label = generate_syn_feature(netG, unseenclasses, class_attributes, opt.syn_num)
        cls = classifier_embed_contras.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), netMap,
                                                  opt.embedSize, data,
                                                  len(unseenclasses), opt.cuda, opt.classifier_lr, 0.5, 100,
                                                  opt.syn_num,
                                                  False)
        acc = cls.acc
        print('unseen class accuracy=%.4f '%acc)
        if acc > best_zsl_acc:
            best_zsl_acc = acc
            torch.save({'state_dict': cls.model.state_dict(), 'epoch': epoch}, f'checkpoints/zsl_best_{acc}.pth')


    # reset G to training mode
    netG.train()
    for p in netMap.parameters():  # reset requires_grad
        p.requires_grad = True
    del train_feature, train_label


from plot import plot_acc, plot_gan_losses, plot_confusion_matrix
from arguments import parse_args
import random
import torch
import torch.backends.cudnn as cudnn
import os
import numpy as np
from dataset import FeaturesCls, FeaturesGAN
from train_cls_65_80 import TrainCls
from train_gan_zsdscr import TrainGAN
from generate import load_unseen_att, load_all_att, load_zero_shot_att, load_few_zero_shot_att, load_seen_att
from splits import get_unseen_class_labels, get_asd_zero_shot_class_labels
import ipdb 
import datetime

opt = parse_args()
dt = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
print(opt.outname)
opt.outname = os.path.join(opt.outname, dt)
print(opt.outname)




if opt.manualSeed is None:
    # opt.manualSeed = random.randint(1, 10000)
    opt.manualSeed = 42

for arg in vars(opt): print(f"######################  {arg}: {getattr(opt, arg)}")


print("Random Seed: ", opt.manualSeed)
seed = opt.manualSeed
torch.manual_seed(seed) # 为CPU设置随机种子
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.	
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
if opt.cuda:
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU，为所有GPU设置随机种子

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# att_labels [ 4, 15, 28, 29, 48, 61, 64], 在原始COCO类别顺序下的编号，没有背景类别
if opt.traincls_classifier == 'zero_shot':
    att_attributes, att_labels = load_zero_shot_att(opt)  
elif opt.traincls_classifier == 'few_zero_shot':
    unseen_attributes, unseen_att_labels = load_few_zero_shot_att(opt)  
elif opt.traincls_classifier == 'base_few_zero_shot':
    att_attributes, att_labels = load_all_att(opt)  

all_attributes, all_att_labels = load_all_att(opt)
seen_attributes, seen_attr_labels = load_seen_att(opt)
unseen_attributes, unseen_att_labels = load_few_zero_shot_att(opt)  
# init classifier

trainCls = TrainCls(opt)

print('initializing GAN Trainer')

start_epoch = 0


seenDataset = FeaturesGAN(opt)


# trainFGGAN = TrainGAN(opt, attributes, att_attributes, att_labels, seen_feats_mean=seenDataset.features_mean, gen_type='FG')
trainFGGAN = TrainGAN(opt, all_attributes, unseen_attributes, unseen_att_labels, seen_attributes, seen_attr_labels, seen_feats_mean=seenDataset.features_mean, gen_type='FG')


try:
    os.makedirs(opt.outname)
except OSError:
    pass

if opt.netD and opt.netG:
    start_epoch = trainFGGAN.load_checkpoint()
    
for epoch in range(start_epoch, opt.nepoch):
    # features, labels = seenDataset.epochData(include_bg=False)
    features, labels = seenDataset.epochData(include_bg=True) # labels 0~65
    # train GAN
    trainFGGAN(epoch, features, labels)
    # synthesize features
    syn_feature, syn_label = trainFGGAN.generate_syn_feature(all_att_labels[:-1], all_attributes[:-1], num=opt.syn_num)
    num_of_bg = opt.syn_num*2
    real_feature_bg, real_label_bg = seenDataset.getBGfeats(num_of_bg)

    # concatenate synthesized + real bg features
    syn_feature = np.concatenate((syn_feature.data.numpy(), real_feature_bg))
    syn_label = np.concatenate((syn_label.data.numpy(), real_label_bg))
    # ipdb.set_trace()
    trainCls(syn_feature, syn_label, gan_epoch=epoch)

    # -----------------------------------------------------------------------------------------------------------------------
    # plots
    # classes = np.concatenate((get_unseen_class_labels(opt.dataset, split=opt.classes_split), ['background']))
    # plot_confusion_matrix(np.load(f'{opt.outname}/confusion_matrix_Train.npy'), classes, classes, opt, dataset='Train', prefix=opt.class_embedding.split('/')[-1])
    # plot_confusion_matrix(np.load(f'{opt.outname}/confusion_matrix_Test.npy'), classes, classes, opt, dataset='Test', prefix=opt.class_embedding.split('/')[-1])
    # plot_acc(np.vstack(trainCls.val_accuracies), opt, prefix=opt.class_embedding.split('/')[-1])

    # save models
    if trainCls.isBestIter == True:
        trainFGGAN.save_checkpoint(state='best')

    trainFGGAN.save_checkpoint(state='latest')




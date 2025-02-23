from __future__ import print_function
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable
import math
from util_48_17 import *
import model_48_17 as model
from cls_models_48_17 import ClsModel, ClsUnseen, Regressor
import ipdb 
import numpy as np
from triplet_util import *
class TrainGAN():
    def __init__(self, opt, attributes, unseenAtt, unseenLabels, seen_attributes, seen_attr_labels, seen_feats_mean, gen_type='FG'):
        
        '''
        CLSWGAN trainer
        Inputs:
            opt -- arguments
            unseenAtt -- embeddings vector of unseen classes
            unseenLabels -- labels of unseen classes
            attributes -- embeddings vector of all classes
        '''
        self.opt = opt

        self.gen_type = gen_type
        
        # 将6个零样本类的COCO id重新编号为0~5
        self.Wu_Labels = np.array([i for i, l in enumerate(unseenLabels)])
        print(f"Wu_Labels {self.Wu_Labels}")
        self.Wu = unseenAtt

        self.Ws_Labels = np.array([i for i, l in enumerate(seen_attr_labels)])
        print(f"Ws_Labels {self.Ws_Labels}")
        self.Ws = seen_attributes

        self.regressor = Regressor(out_sz=self.Wu.shape[1]) # 构建unseen的分类器，先预测语义，然后再匹配unseen类别的语义表示得到最终的概率结果，分类的类别数为7，不包含背景类
        self.regressor.cuda()
        # MSCOCO/unseen_Classifier.pth
        self.regressor = loadUnseenWeights(opt.pretrain_regressor, self.regressor)
        self.reg_criterion= nn.MSELoss(reduction='sum') 


        self.classifier = ClsModel(num_classes=opt.nclass_all) # 构建81类的分类器，分类模型是一层FC，直接输出最终的概率
        self.classifier.cuda()
        self.classifier = loadFasterRcnnCLSHead(opt.pretrain_classifier, self.classifier) # 加载预训练Seen检测器中的分类器权重做初始化

        # 未知类的分类器和81类的分类器都不需要训练，那损失是怎么反传的？
        for p in self.classifier.parameters():
            p.requires_grad = False
        for p in self.regressor.parameters():
            p.requires_grad = False


        mu_dtilde= opt.tr_mu_dtilde
        sigma_dtilde=opt.tr_sigma_dtilde #standard deviation of flexible margin
       
        self.D_tilde = Variable(distance_matrix(
        attributes.float(),
        mahalanobis=True,
        mean=mu_dtilde, std=mu_dtilde*sigma_dtilde ).cuda())

        print(f"D_tilde , mu_dtilde= {mu_dtilde}, sigma_dtilde= {sigma_dtilde} :{self.D_tilde}")
        
        
        print(f"Triplet_hyperpameter: {opt.triplet_lamda}")
        self.ntrain = opt.gan_epoch_budget
        self.attributes = attributes.data.numpy()

        print(f"# of training samples: {self.ntrain}")
        # initialize generator and discriminator
        self.netG = model.MLP_G(self.opt)
        self.netD = model.MLP_CRITIC(self.opt)
        


        if self.opt.cuda and torch.cuda.is_available():
            self.netG = self.netG.cuda()
            self.netD = self.netD.cuda()

        print('\n\n#############################################################\n')
        print(self.netG, '\n')
        print(self.netD)
        print('\n#############################################################\n\n')

        # classification loss, Equation (4) of the paper
        self.cls_criterion = nn.NLLLoss()

        # self.one = torch.FloatTensor([1])
        self.one = torch.tensor(1, dtype=torch.float)
        self.mone = self.one * -1

        if self.opt.cuda:
            
            self.one = self.one.cuda()
            self.mone = self.mone.cuda()
            self.cls_criterion.cuda()


        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    def __call__(self, epoch, features, labels):
        """
        Train GAN for one epoch
        Inputs:
            epoch: current epoch
            features: current epoch subset of features
            labels: ground truth labels
        """
        self.epoch = epoch
        self.features = features
        self.labels = labels
        self.ntrain = len(self.labels)
        self.trainEpoch()
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.opt.netG)
        self.netG.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']
        self.netD.load_state_dict(torch.load(self.opt.netD)['state_dict'])
        print(f"loaded weights from epoch: {epoch} \n{self.opt.netD} \n{self.opt.netG} \n")
        return epoch
    def save_checkpoint(self, state='latest'):
        torch.save({'state_dict': self.netD.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/disc_{state}.pth')
        torch.save({'state_dict': self.netG.state_dict(), 'epoch': self.epoch}, f'{self.opt.outname}/gen_{state}.pth')

    def generate_syn_feature(self, labels, attribute, num=100, no_grad=True):
        """
        generates features
        inputs:
            labels: features labels to generate nx1 n is number of objects 
            attributes: attributes of objects to generate (nxd) d is attribute dimensions
            num: number of features to generate for each object
        returns:
            1) synthesised features 
            2) labels of synthesised  features 
        """

        nclass = labels.shape[0] # nclass is 7
        syn_feature = torch.FloatTensor(nclass * num , self.opt.resSize) # num=250
        syn_label = torch.LongTensor(nclass*num)

        syn_att = torch.FloatTensor(num, self.opt.attSize)
        syn_noise = torch.FloatTensor(num, self.opt.nz)
        
        if self.opt.cuda:
            syn_att = syn_att.cuda()
            syn_noise = syn_noise.cuda()
        if no_grad is True:
            with torch.no_grad():
                for i in range(nclass):
                    label = labels[i]
                    iclass_att = attribute[i]
                    syn_att.copy_(iclass_att.repeat(num, 1))
                    syn_noise.normal_(0, 1)
                    output = self.netG(Variable(syn_noise), Variable(syn_att))
                    syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
                    syn_label.narrow(0, i*num, num).fill_(label)
        else:
            for i in range(nclass):
                label = labels[i]
                iclass_att = attribute[i]
                syn_att.copy_(iclass_att.repeat(num, 1))
                syn_noise.normal_(0, 1)
                output = self.netG(Variable(syn_noise), Variable(syn_att))
                syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
                syn_label.narrow(0, i*num, num).fill_(label)

        return syn_feature, syn_label

    def sample(self):
        """
        randomaly samples one batch of data
        returns (1)real features, (2)labels (3) attributes embeddings
        """
        idx = torch.randperm(self.ntrain)[0:self.opt.batch_size]
        batch_feature = torch.from_numpy(self.features[idx])
        batch_label = torch.from_numpy(self.labels[idx])
        batch_att = torch.from_numpy(self.attributes[batch_label])
        if 'BG' == self.gen_type:
            batch_label*=0
        return batch_feature, batch_label, batch_att

    def calc_gradient_penalty(self, real_data, fake_data, input_att):
        
        alpha = torch.rand(self.opt.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        if self.opt.cuda:
            alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        if self.opt.cuda:
            interpolates = interpolates.cuda()

        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = self.netD(interpolates, Variable(input_att))

        ones = torch.ones(disc_interpolates.size())
        if self.opt.cuda:
            ones = ones.cuda()

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=ones,
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.opt.lambda1
        return gradient_penalty

    def get_z_random(self):
        """
        returns normal initialized noise tensor 
        """
        z = torch.cuda.FloatTensor(self.opt.batch_size, self.opt.nz)
        z.normal_(0, 1)
        return z

    def trainEpoch(self):

        for i in range(0, self.ntrain, self.opt.batch_size):
            # import ipdb; ipdb.set_trace()
            input_res, input_label, input_att = self.sample()

            if self.opt.batch_size != input_res.shape[0]:
                continue
            input_res, input_label, input_att = input_res.type(torch.FloatTensor).cuda(), input_label.type(torch.LongTensor).cuda(), input_att.type(torch.FloatTensor).cuda()
            ############################
            # (1) Update D network: optimize WGAN-GP objective, Equation (2)
            ###########################
            for p in self.netD.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            for iter_d in range(self.opt.critic_iter):
                self.netD.zero_grad()
                # train with realG
                # sample a mini-batch
                # sparse_real = self.opt.resSize - input_res[1].gt(0).sum()
                input_resv = Variable(input_res)
                input_attv = Variable(input_att)
                
                criticD_real = self.netD(input_resv, input_attv)
                criticD_real = criticD_real.mean()
                # ipdb.set_trace()
                criticD_real.backward(self.mone)

                # train with fakeG
                # noise.normal_(0, 1)
                noise = self.get_z_random()

                noisev = Variable(noise)
                fake = self.netG(noisev, input_attv)
                criticD_fake = self.netD(fake.detach(), input_attv)
                criticD_fake = criticD_fake.mean()
                criticD_fake.backward(self.one)

                # gradient penalty
                gradient_penalty = self.calc_gradient_penalty(input_res, fake.data, input_att)
                gradient_penalty.backward()


                # unseenc_errG.backward()

                Wasserstein_D = criticD_real - criticD_fake
                D_cost = criticD_fake - criticD_real + gradient_penalty
                # D_cost.backward()

                self.optimizerD.step()

            ############################
            # (2) Update G network: optimize WGAN-GP objective, Equation (2)
            ###########################
            for p in self.netD.parameters(): # reset requires_grad
                p.requires_grad = False # avoid computation

            self.netG.zero_grad()
            input_attv = Variable(input_att)
            noise = self.get_z_random()
            noisev = Variable(noise)
            fake = self.netG(noisev, input_attv)
            criticG_fake = self.netD(fake, input_attv)
            criticG_fake = criticG_fake.mean()
            G_cost = criticG_fake

            # ---------------------
            # mode seeking loss https://github.com/HelenMao/MSGAN/blob/e386c252f059703fcf5d439258949ae03dc46519/DCGAN-Mode-Seeking/model.py#L66
            noise2 = self.get_z_random()

            noise2v = Variable(noise2)
            fake2 = self.netG(noise2v, input_attv)

            lz = torch.mean(torch.abs(fake2 - fake)) / torch.mean(
                torch.abs(noise2v - noisev))
            eps = 1 * 1e-5
            loss_lz = 1 / (lz + eps)
            loss_lz*=self.opt.lz_ratio
            # ---------------------

            # classification loss
            # seen
            c_errG = self.cls_criterion(self.classifier(feats=fake, classifier_only=True), Variable(input_label))
            c_errG = self.opt.cls_weight*c_errG
            # --------------------------------------------

            # unseen 
            fake_unseen_feat, fake_unseen_lab = self.generate_syn_feature(self.Wu_Labels, self.Wu, num=self.opt.batch_size//4, no_grad=False)
            pred_unseen_semantic = self.regressor(feats=fake_unseen_feat.float().cuda())
            true_unseen_semantic = self.Wu[fake_unseen_lab]
            # ipdb.set_trace()
            unseen_cyc_err =  self.reg_criterion(pred_unseen_semantic.float().cuda(),true_unseen_semantic.float().cuda() )
            # seen
            noise4 = self.get_z_random()
            noise4v = Variable(noise4)
            fake4 = self.netG(noise4v, input_attv)
            pred_seen_semantic=self.regressor(feats=fake4.cuda())
            
            seen_cyc_err = self.reg_criterion(pred_seen_semantic.float().cuda(),input_attv.float().cuda() )
            cyc_err = self.opt.regressor_lamda * ( seen_cyc_err+ unseen_cyc_err)


            ## triplet loss
            noise3 = self.get_z_random()
            noise3v = Variable(noise3)
            fake_generated_seen = self.netG(noise3v, input_attv)
            triplet_selector= AllTripletSelector()
            # seen
            t_loss=TripletLoss(self.D_tilde,input_label,triplet_selector) # variable margin
            t_loss_seen=t_loss.forward(fake_generated_seen,input_label)
            # unseen
            fake_unseen_feat2, fake_unseen_lab2 = self.generate_syn_feature(self.Wu_Labels, self.Wu, num=self.opt.batch_size//16, no_grad=False)
            t_loss_u=TripletLoss(self.D_tilde,fake_unseen_lab2,triplet_selector) #var marg
            t_loss_unseen=t_loss_u.forward(fake_unseen_feat2.float().cuda(),fake_unseen_lab2.cuda())
            cal_triplet_loss=self.opt.triplet_lamda*(t_loss_seen + t_loss_unseen)
            # ---------------------------------------------
            # Total loss 

            errG = -G_cost + c_errG + loss_lz + cyc_err + cal_triplet_loss
            errG.backward()
            self.optimizerG.step()

            print(f"{self.gen_type} [{self.epoch+1:02}/{self.opt.nepoch:02}] [{i:06}/{int(self.ntrain)}] Loss: {errG.item() :0.4f} D loss: {D_cost.data.item():.4f} G loss: {G_cost.data.item():.4f}, W dist: {Wasserstein_D.data.item():.4f} seen loss: {c_errG.data.item():.4f}  regressor loss: {cyc_err.data.item():.4f} loss div: {loss_lz.item():0.4f}   triplet loss: {cal_triplet_loss.data.item():.4f}")
            
        self.netG.eval()

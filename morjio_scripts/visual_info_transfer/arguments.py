import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='coco', help='coco, voc')
    parser.add_argument('--dataroot', default='../data', help='path to dataset')
    parser.add_argument('--class_embedding', default='VOC/fasttext_multilabels_4.npy')
    parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
    
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
    parser.add_argument('--resSize', type=int, default=1024, help='size of visual features')
    parser.add_argument('--attSize', type=int, default=300, help='size of semantic features')
    parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
    parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
    parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train GAN')
    parser.add_argument('--nepoch_cls', type=int, default=2000, help='number of epochs to train CLS')
    parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
    parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    parser.add_argument('--cls_weight', type=float, default=0.001, help='weight of the classification loss')
    parser.add_argument('--cls_weight_zero_shot', type=float, default=0.001, help='weight of the classification loss')
    parser.add_argument('--cls_weight_unseen', type=float, default=0.001, help='weight of the classification loss')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
    parser.add_argument('--lr_cls', type=float, default=0.0001, help='learning rate to train CLS ')
    parser.add_argument('--testsplit', default='test', help='unseen classes feats and labels paths')
    parser.add_argument('--trainsplit', default='train', help='seen classes feats and labels paths')

    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lz_ratio', type=float, default=1.0, help='mode seeking loss weight')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (for seen classes loss on fake features)")
    parser.add_argument('--pretrain_classifier_zero_shot', default='', help="path to pretrain classifier (for unseen classes loss on fake features)")
    parser.add_argument('--pretrain_classifier_unseen', default='', help="path to pretrain classifier (for unseen classes loss on fake features)")
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--netG_name', default='')
    parser.add_argument('--netD_name', default='')
    parser.add_argument('--classes_split', default='')
    parser.add_argument('--outname', default='./checkpoints/', help='folder to output data and model checkpoints')
    parser.add_argument('--val_every', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nclass_all', type=int, default=81, help='number of all classes')
    parser.add_argument('--lr_step', type=int, default=30, help='number of all classes')
    parser.add_argument('--gan_epoch_budget', type=int, default=10000, help='random pick subset of features to train GAN')
    parser.add_argument('--traincls_classifier', default='zero_shot', help='zero_shot, few_zero_shot, base_few_zero_shot')
    parser.add_argument('--supconloss', action='store_true', default=False, help='use supervised constrastive loss')

    # by ce-gzsl
    parser.add_argument('--ins_weight', type=float, default=0.001, help='weight of the classification loss when learning G')
    parser.add_argument('--ins_temp', type=float, default=0.1, help='temperature in instance-level supervision')
    parser.add_argument('--embedSize', type=int, default=2048, help='size of embedding h')
    parser.add_argument('--outzSize', type=int, default=512, help='size of non-liner projection z')
    parser.add_argument('--nhF', type=int, default=2048, help='size of the hidden units comparator network F')

    ## zsdscr
    parser.add_argument('--regressor_lamda', type=float, default=0.01, help='')
    parser.add_argument('--triplet_lamda', type=float, default=0.1, help='')
    parser.add_argument('--pretrain_regressor',  default='data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/regressor_tfa.pth', help='')
    parser.add_argument('--tr_mu_dtilde', type=float, default=0.5, help='')
    parser.add_argument('--tr_sigma_dtilde', type=float, default=0.5, help='')

    ## mixup 
    parser.add_argument('--mixup', action='store_true', default=False, help='use mixup')

    ## 65_80
    parser.add_argument('--testcls_classifier', default='few_zero_shot', help='zero_shot, few_zero_shot, base_few_zero_shot')

    ## voc split
    parser.add_argument('--voc_split', default='split1', help='split1~3')

    opt = parser.parse_args()
    return opt
CUDA_VISIBLE_DEVICES=3 python morjio_scripts/suzsd_with_classifier_fusion/trainer.py --manualSeed 42 \
--cls_weight 0.001 --cls_weight_zero_shot 0.001 --nclass_all 81 --syn_num 250 --val_every 1 \
--cuda --netG_name MLP_G --netD_name MLP_D \
--nepoch 100 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--dataset coco --batch_size 32 --nz 300 --attSize 300 --resSize 1024 --gan_epoch_budget 38000 \
--lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
--pretrain_classifier work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth \
--pretrain_classifier_zero_shot data/coco/any_shot_detection/unseen_Classifier.pth \
--class_embedding data/coco/any_shot_detection/fasttext_switch_bg.npy \
--dataroot data/coco/any_shot_detection/base_few_shot_det/multi_labels/zero_shot  \
--testsplit test_0.6_0.3 \
--trainsplit train_0.6_0.3 \
--classes_split 65_8_7 \
--lz_ratio 0.01 \
--outname checkpoints/asd_65_8_7 \
--traincls_classifier zero_shot \
# --supconloss


# pretrain_classifier  预训练的检测模型，用于提取其对应的分类器
# pretrain_classifier_unseen  预训练的unseen分类器
# classes_split 数据划分
# checkpoints 基于生成特征训练的unseen分类模型
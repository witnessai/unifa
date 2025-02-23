## text embedding，同时迁移regressor和triplet loss， mixup, tfa, 10shot
## 改变shot数量或者tfa方法，只需要改变pretrain_classifier、dataroot、outname 三个参数
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/visual_info_transfer/trainer_zsdscr.py --manualSeed 42 \
# --cls_weight 0.01 --cls_weight_unseen 0 --nclass_all 81 --syn_num 250 --val_every 1 \
# --cuda --netG_name MLP_G --netD_name MLP_D \
# --nepoch 150 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
# --dataset coco --batch_size 128 --nz 300 --attSize 512 --resSize 1024 --gan_epoch_budget 38000 \
# --lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
# --pretrain_classifier work_dirs/tfa_r101_fpn_coco_10shot-fine-tuning_34server/iter_160000.pth \
# --pretrain_classifier_unseen data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/pretrained_classifier/unseen_Classifier_textemb.pth \
# --class_embedding data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_with_bg_switch_bg_for_mmdet2.npy \
# --dataroot data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/10shot  \
# --testsplit test_0.6_0.3 \
# --trainsplit train_0.6_0.3 \
# --classes_split 60_20 \
# --lz_ratio 0.01 \
# --outname checkpoints/fsd_60_20/both_regressor_triplet/tfa/10shot \
# --traincls_classifier few_zero_shot \
# --regressor_lamda 0.01 \
# --triplet_lamda 0.1 \
# --pretrain_regressor data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/regressor_textemb_tfa.pth --tr_mu_dtilde 0.5 --tr_sigma_dtilde 0.5 \
# --mixup


## text embedding，同时迁移regressor和triplet loss， mixup, tfa, 30shot
## 改变shot数量或者tfa方法，只需要改变pretrain_classifier、dataroot、outname 三个参数
# CUDA_VISIBLE_DEVICES=2 python morjio_scripts/visual_info_transfer/trainer_zsdscr.py --manualSeed 42 \
# --cls_weight 0.01 --cls_weight_unseen 0 --nclass_all 81 --syn_num 250 --val_every 1 \
# --cuda --netG_name MLP_G --netD_name MLP_D \
# --nepoch 150 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
# --dataset coco --batch_size 128 --nz 300 --attSize 512 --resSize 1024 --gan_epoch_budget 38000 \
# --lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
# --pretrain_classifier work_dirs/tfa_r101_fpn_coco_30shot-fine-tuning_34server/iter_240000.pth \
# --pretrain_classifier_unseen data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/pretrained_classifier/unseen_Classifier_textemb.pth \
# --class_embedding data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_with_bg_switch_bg_for_mmdet2.npy \
# --dataroot data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/30shot  \
# --testsplit test_0.6_0.3 \
# --trainsplit train_0.6_0.3 \
# --classes_split 60_20 \
# --lz_ratio 0.01 \
# --outname checkpoints/fsd_60_20/both_regressor_triplet/tfa/30shot \
# --traincls_classifier few_zero_shot \
# --regressor_lamda 0.01 \
# --triplet_lamda 0.1 \
# --pretrain_regressor data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/regressor_textemb_tfa.pth --tr_mu_dtilde 0.5 --tr_sigma_dtilde 0.5 \
# --mixup


## text embedding，同时迁移regressor和triplet loss， mixup, fsce, 10shot
## 改变shot数量或者tfa方法，只需要改变pretrain_classifier、dataroot、outname 三个参数
# CUDA_VISIBLE_DEVICES=2 python morjio_scripts/visual_info_transfer/trainer_zsdscr.py --manualSeed 42 \
# --cls_weight 0.01 --cls_weight_unseen 0 --nclass_all 81 --syn_num 250 --val_every 1 \
# --cuda --netG_name MLP_G --netD_name MLP_D \
# --nepoch 150 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
# --dataset coco --batch_size 128 --nz 300 --attSize 512 --resSize 1024 --gan_epoch_budget 38000 \
# --lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
# --pretrain_classifier work_dirs/fsce_r101_fpn_coco_10shot-fine-tuning_34server/iter_30000.pth \
# --pretrain_classifier_unseen data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/pretrained_classifier/unseen_Classifier_textemb.pth \
# --class_embedding data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_with_bg_switch_bg_for_mmdet2.npy \
# --dataroot data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/10shot  \
# --testsplit test_0.6_0.3 \
# --trainsplit train_0.6_0.3 \
# --classes_split 60_20 \
# --lz_ratio 0.01 \
# --outname checkpoints/fsd_60_20/both_regressor_triplet/fsce/10shot \
# --traincls_classifier few_zero_shot \
# --regressor_lamda 0.01 \
# --triplet_lamda 0.1 \
# --pretrain_regressor data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/regressor_textemb_tfa.pth --tr_mu_dtilde 0.5 --tr_sigma_dtilde 0.5 \
# --mixup



## text embedding，同时迁移regressor和triplet loss， mixup, fsce, 30shot
## 改变shot数量或者tfa方法，只需要改变pretrain_classifier、dataroot、outname 三个参数
CUDA_VISIBLE_DEVICES=0 python morjio_scripts/visual_info_transfer/trainer_zsdscr.py --manualSeed 42 \
--cls_weight 0.01 --cls_weight_unseen 0 --nclass_all 81 --syn_num 250 --val_every 1 \
--cuda --netG_name MLP_G --netD_name MLP_D \
--nepoch 150 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--dataset coco --batch_size 128 --nz 300 --attSize 512 --resSize 1024 --gan_epoch_budget 38000 \
--lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
--pretrain_classifier work_dirs/fsce_r101_fpn_coco_30shot-fine-tuning_34server/iter_40000.pth \
--pretrain_classifier_unseen data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/pretrained_classifier/unseen_Classifier_textemb.pth \
--class_embedding data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_with_bg_switch_bg_for_mmdet2.npy \
--dataroot data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/30shot  \
--testsplit test_0.6_0.3 \
--trainsplit train_0.6_0.3 \
--classes_split 60_20 \
--lz_ratio 0.01 \
--outname checkpoints/fsd_60_20/both_regressor_triplet/fsce/30shot \
--traincls_classifier few_zero_shot \
--regressor_lamda 0.01 \
--triplet_lamda 0.1 \
--pretrain_regressor data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/regressor_textemb_tfa.pth --tr_mu_dtilde 0.5 --tr_sigma_dtilde 0.5 \
--mixup
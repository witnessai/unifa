## text embedding，同时迁移regressor和triplet loss， mixup
# CUDA_VISIBLE_DEVICES=2 python morjio_scripts/visual_info_transfer/trainer_zsdscr.py --manualSeed 42 \
# --cls_weight 0.01 --cls_weight_unseen 0 --nclass_all 81 --syn_num 250 --val_every 1 \
# --cuda --netG_name MLP_G --netD_name MLP_D \
# --nepoch 150 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
# --dataset coco --batch_size 128 --nz 300 --attSize 512 --resSize 1024 --gan_epoch_budget 38000 \
# --lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
# --pretrain_classifier work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth \
# --pretrain_classifier_unseen data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/pretrained_classifier/unseen_Classifier_textemb.pth \
# --class_embedding data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_with_bg_switch_bg_for_mmdet2.npy \
# --dataroot data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/tfa/10shot  \
# --testsplit test_0.6_0.3 \
# --trainsplit train_0.6_0.3 \
# --classes_split 65_15 \
# --lz_ratio 0.01 \
# --outname checkpoints/fsd_65_15/both_regressor_triplet \
# --traincls_classifier few_zero_shot \
# --regressor_lamda 0.01 \
# --triplet_lamda 0.1 \
# --pretrain_regressor data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/regressor_textemb_tfa.pth --tr_mu_dtilde 0.5 --tr_sigma_dtilde 0.5 \
# --mixup

## text embedding，同时迁移regressor和triplet loss
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/visual_info_transfer/trainer_zsdscr.py --manualSeed 42 \
# --cls_weight 0.01 --cls_weight_unseen 0 --nclass_all 81 --syn_num 250 --val_every 1 \
# --cuda --netG_name MLP_G --netD_name MLP_D \
# --nepoch 150 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
# --dataset coco --batch_size 128 --nz 300 --attSize 512 --resSize 1024 --gan_epoch_budget 38000 \
# --lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
# --pretrain_classifier work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth \
# --pretrain_classifier_unseen data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/pretrained_classifier/unseen_Classifier_textemb.pth \
# --class_embedding data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_with_bg_switch_bg_for_mmdet2.npy \
# --dataroot data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/tfa/10shot  \
# --testsplit test_0.6_0.3 \
# --trainsplit train_0.6_0.3 \
# --classes_split 65_15 \
# --lz_ratio 0.01 \
# --outname checkpoints/fsd_65_15/both_regressor_triplet \
# --traincls_classifier few_zero_shot \
# --regressor_lamda 0.01 \
# --triplet_lamda 0.1 \
# --pretrain_regressor data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/regressor_textemb_tfa.pth --tr_mu_dtilde 0.5 --tr_sigma_dtilde 0.5 



## text embedding, 只迁移regressor loss
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/visual_info_transfer/trainer_zsdscr.py --manualSeed 42 \
# --cls_weight 0.01 --cls_weight_unseen 0 --nclass_all 81 --syn_num 250 --val_every 1 \
# --cuda --netG_name MLP_G --netD_name MLP_D \
# --nepoch 100 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
# --dataset coco --batch_size 128 --nz 300 --attSize 512 --resSize 1024 --gan_epoch_budget 38000 \
# --lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
# --pretrain_classifier work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth \
# --pretrain_classifier_unseen data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/pretrained_classifier/unseen_Classifier_textemb.pth \
# --class_embedding data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_with_bg_switch_bg_for_mmdet2.npy \
# --dataroot data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/tfa/10shot  \
# --testsplit test_0.6_0.3 \
# --trainsplit train_0.6_0.3 \
# --classes_split 65_15 \
# --lz_ratio 0.01 \
# --outname checkpoints/fsd_65_15/only_regressor/ \
# --traincls_classifier few_zero_shot \
# --regressor_lamda 0.01 \
# --triplet_lamda 0.0 \
# --pretrain_regressor data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/regressor_textemb_tfa.pth --tr_mu_dtilde 0.5 --tr_sigma_dtilde 0.5 

## text embedding, 只迁移triplet loss
# CUDA_VISIBLE_DEVICES=2 python morjio_scripts/visual_info_transfer/trainer_zsdscr.py --manualSeed 42 \
# --cls_weight 0.01 --cls_weight_unseen 0 --nclass_all 81 --syn_num 250 --val_every 1 \
# --cuda --netG_name MLP_G --netD_name MLP_D \
# --nepoch 100 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
# --dataset coco --batch_size 128 --nz 300 --attSize 512 --resSize 1024 --gan_epoch_budget 38000 \
# --lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
# --pretrain_classifier work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth \
# --pretrain_classifier_unseen data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/pretrained_classifier/unseen_Classifier_textemb.pth \
# --class_embedding data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_with_bg_switch_bg_for_mmdet2.npy \
# --dataroot data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/tfa/10shot  \
# --testsplit test_0.6_0.3 \
# --trainsplit train_0.6_0.3 \
# --classes_split 65_15 \
# --lz_ratio 0.01 \
# --outname checkpoints/fsd_65_15/only_triplet/ \
# --traincls_classifier few_zero_shot \
# --regressor_lamda 0.00 \
# --triplet_lamda 0.1 \
# --pretrain_regressor data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/regressor_textemb_tfa.pth --tr_mu_dtilde 0.5 --tr_sigma_dtilde 0.5 



## word embedding，同时迁移regressor和triplet loss
# CUDA_VISIBLE_DEVICES=1 python morjio_scripts/visual_info_transfer/trainer_zsdscr.py --manualSeed 42 \
# --cls_weight 0.01 --cls_weight_unseen 0 --nclass_all 81 --syn_num 250 --val_every 1 \
# --cuda --netG_name MLP_G --netD_name MLP_D \
# --nepoch 100 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
# --dataset coco --batch_size 128 --nz 300 --attSize 300 --resSize 1024 --gan_epoch_budget 38000 \
# --lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
# --pretrain_classifier work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth \
# --pretrain_classifier_unseen data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/unseen_Classifier.pth \
# --class_embedding data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/fasttext_switch_bg.npy \
# --dataroot data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/tfa/10shot  \
# --testsplit test_0.6_0.3 \
# --trainsplit train_0.6_0.3 \
# --classes_split 65_15 \
# --lz_ratio 0.01 \
# --outname checkpoints/fsd_65_15/wordemb_zsdscr \
# --traincls_classifier few_zero_shot \
# --regressor_lamda 0.01 \
# --triplet_lamda 0.1 \
# --pretrain_regressor data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/regressor_wordemb_tfa.pth --tr_mu_dtilde 0.5 --tr_sigma_dtilde 0.5


## text embedding，同时迁移regressor和triplet loss， mixup, 生成特征由250增长至1000:无效果
# CUDA_VISIBLE_DEVICES=2 python morjio_scripts/visual_info_transfer/trainer_zsdscr.py --manualSeed 42 \
# --cls_weight 0.01 --cls_weight_unseen 0 --nclass_all 81 --syn_num 1000 --val_every 1 \
# --cuda --netG_name MLP_G --netD_name MLP_D \
# --nepoch 150 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
# --dataset coco --batch_size 128 --nz 300 --attSize 512 --resSize 1024 --gan_epoch_budget 38000 \
# --lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
# --pretrain_classifier work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth \
# --pretrain_classifier_unseen data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/pretrained_classifier/unseen_Classifier_textemb.pth \
# --class_embedding data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_with_bg_switch_bg_for_mmdet2.npy \
# --dataroot data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/tfa/10shot  \
# --testsplit test_0.6_0.3 \
# --trainsplit train_0.6_0.3 \
# --classes_split 65_15 \
# --lz_ratio 0.01 \
# --outname checkpoints/fsd_65_15/both_regressor_triplet \
# --traincls_classifier few_zero_shot \
# --regressor_lamda 0.01 \
# --triplet_lamda 0.1 \
# --pretrain_regressor data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/regressor_textemb_tfa.pth --tr_mu_dtilde 0.5 --tr_sigma_dtilde 0.5 \
# --mixup
## text embedding，同时迁移regressor和triplet loss， mixup, 5shot, fsce add cpe
CUDA_VISIBLE_DEVICES=4 python morjio_scripts/visual_info_transfer/trainer_zsdscr_voc_15_5.py --manualSeed 42 \
--cls_weight 0.01 --cls_weight_unseen 0 --nclass_all 21 --syn_num 250 --val_every 1 \
--cuda --netG_name MLP_G --netD_name MLP_D \
--nepoch 150 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--dataset voc --batch_size 128 --nz 300 --attSize 512 --resSize 1024 --gan_epoch_budget 38000 \
--lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
--pretrain_classifier work_dirs/fsce_r101_fpn_voc-split1_base-training/base_model_random_init_bbox_head.pth \
--pretrain_classifier_unseen data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/pretrained_classifier/unseen_Classifier_textemb.pth \
--class_embedding data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/class_embedding/voc_all_classnames_split1_text_embedding_use_prompts_7_base_novel_order_with_bg_switch_bg_for_mmdet2.npy \
--dataroot data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce_add_cpe/5shot  \
--testsplit test_0.6_0.3 \
--trainsplit train_0.6_0.3 \
--classes_split 15_5_1 \
--lz_ratio 0.01 \
--outname checkpoints/fsd_15_5/both_regressor_triplet/fsce_add_cpe/5shot \
--traincls_classifier few_zero_shot \
--regressor_lamda 0.01 \
--triplet_lamda 0.1 \
--pretrain_regressor data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/regressor_textemb_tfa.pth --tr_mu_dtilde 0.5 --tr_sigma_dtilde 0.5 \
--mixup \
--voc_split split1

## text embedding，同时迁移regressor和triplet loss， mixup, 5shot, fsce 
# CUDA_VISIBLE_DEVICES=2 python morjio_scripts/visual_info_transfer/trainer_zsdscr_voc_15_5.py --manualSeed 42 \
# --cls_weight 0.01 --cls_weight_unseen 0 --nclass_all 21 --syn_num 250 --val_every 1 \
# --cuda --netG_name MLP_G --netD_name MLP_D \
# --nepoch 150 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
# --dataset voc --batch_size 128 --nz 300 --attSize 512 --resSize 1024 --gan_epoch_budget 38000 \
# --lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
# --pretrain_classifier work_dirs/fsce_r101_fpn_voc-split1_base-training/base_model_random_init_bbox_head.pth \
# --pretrain_classifier_unseen data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/pretrained_classifier/unseen_Classifier_textemb.pth \
# --class_embedding data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/class_embedding/voc_all_classnames_split1_text_embedding_use_prompts_7_base_novel_order_with_bg_switch_bg_for_mmdet2.npy \
# --dataroot data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce/5shot  \
# --testsplit test_0.6_0.3 \
# --trainsplit train_0.6_0.3 \
# --classes_split 15_5_1 \
# --lz_ratio 0.01 \
# --outname checkpoints/fsd_15_5/both_regressor_triplet/fsce/5shot \
# --traincls_classifier few_zero_shot \
# --regressor_lamda 0.01 \
# --triplet_lamda 0.1 \
# --pretrain_regressor data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/regressor_textemb_tfa.pth --tr_mu_dtilde 0.5 --tr_sigma_dtilde 0.5 \
# --mixup \
# --voc_split split1



## text embedding，同时迁移regressor和triplet loss， mixup, 3shot, fsce 
# CUDA_VISIBLE_DEVICES=1 python morjio_scripts/visual_info_transfer/trainer_zsdscr_voc_15_5.py --manualSeed 42 \
# --cls_weight 0.01 --cls_weight_unseen 0 --nclass_all 21 --syn_num 250 --val_every 1 \
# --cuda --netG_name MLP_G --netD_name MLP_D \
# --nepoch 150 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
# --dataset voc --batch_size 128 --nz 300 --attSize 512 --resSize 1024 --gan_epoch_budget 38000 \
# --lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
# --pretrain_classifier work_dirs/fsce_r101_fpn_voc-split1_base-training/base_model_random_init_bbox_head.pth \
# --pretrain_classifier_unseen data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/pretrained_classifier/unseen_Classifier_textemb.pth \
# --class_embedding data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/class_embedding/voc_all_classnames_split1_text_embedding_use_prompts_7_base_novel_order_with_bg_switch_bg_for_mmdet2.npy \
# --dataroot data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce/3shot  \
# --testsplit test_0.6_0.3 \
# --trainsplit train_0.6_0.3 \
# --classes_split 15_5_1 \
# --lz_ratio 0.01 \
# --outname checkpoints/fsd_15_5/both_regressor_triplet/fsce/3shot \
# --traincls_classifier few_zero_shot \
# --regressor_lamda 0.01 \
# --triplet_lamda 0.1 \
# --pretrain_regressor data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/regressor_textemb_tfa.pth --tr_mu_dtilde 0.5 --tr_sigma_dtilde 0.5 \
# --mixup \
# --voc_split split1



## text embedding，同时迁移regressor和triplet loss， mixup, 5shot
# CUDA_VISIBLE_DEVICES=2 python morjio_scripts/visual_info_transfer/trainer_zsdscr_voc_15_5.py --manualSeed 42 \
# --cls_weight 0.01 --cls_weight_unseen 0 --nclass_all 21 --syn_num 250 --val_every 1 \
# --cuda --netG_name MLP_G --netD_name MLP_D \
# --nepoch 150 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
# --dataset voc --batch_size 128 --nz 300 --attSize 512 --resSize 1024 --gan_epoch_budget 38000 \
# --lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
# --pretrain_classifier work_dirs/tfa_r101_fpn_voc-split1_base-training/base_model_random_init_bbox_head.pth \
# --pretrain_classifier_unseen data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/pretrained_classifier/unseen_Classifier_textemb.pth \
# --class_embedding data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/class_embedding/voc_all_classnames_split1_text_embedding_use_prompts_7_base_novel_order_with_bg_switch_bg_for_mmdet2.npy \
# --dataroot data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/5shot  \
# --testsplit test_0.6_0.3 \
# --trainsplit train_0.6_0.3 \
# --classes_split 15_5_1 \
# --lz_ratio 0.01 \
# --outname checkpoints/fsd_15_5/both_regressor_triplet/5shot \
# --traincls_classifier few_zero_shot \
# --regressor_lamda 0.01 \
# --triplet_lamda 0.1 \
# --pretrain_regressor data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/regressor_textemb_tfa.pth --tr_mu_dtilde 0.5 --tr_sigma_dtilde 0.5 \
# --mixup \
# # --voc_split split1

## text embedding，同时迁移regressor和triplet loss， mixup, 3shot
## classes_split: 15_5_1, means base/novel split is 15_5, and voc_split is 1
# CUDA_VISIBLE_DEVICES=1 python morjio_scripts/visual_info_transfer/trainer_zsdscr_voc_15_5.py --manualSeed 42 \
# --cls_weight 0.01 --cls_weight_unseen 0 --nclass_all 21 --syn_num 250 --val_every 1 \
# --cuda --netG_name MLP_G --netD_name MLP_D \
# --nepoch 150 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
# --dataset voc --batch_size 128 --nz 300 --attSize 512 --resSize 1024 --gan_epoch_budget 38000 \
# --lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
# --pretrain_classifier work_dirs/tfa_r101_fpn_voc-split1_base-training/base_model_random_init_bbox_head.pth \
# --pretrain_classifier_unseen data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/pretrained_classifier/unseen_Classifier_textemb.pth \
# --class_embedding data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/class_embedding/voc_all_classnames_split1_text_embedding_use_prompts_7_base_novel_order_with_bg_switch_bg_for_mmdet2.npy \
# --dataroot data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/3shot  \
# --testsplit test_0.6_0.3 \
# --trainsplit train_0.6_0.3 \
# --classes_split 15_5_1 \
# --lz_ratio 0.01 \
# --outname checkpoints/fsd_15_5/both_regressor_triplet/3shot \
# --traincls_classifier few_zero_shot \
# --regressor_lamda 0.01 \
# --triplet_lamda 0.1 \
# --pretrain_regressor data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/regressor_textemb_tfa.pth --tr_mu_dtilde 0.5 --tr_sigma_dtilde 0.5 \
# --mixup \
# --voc_split split1





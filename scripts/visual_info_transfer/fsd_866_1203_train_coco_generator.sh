## CLIP text embedding, increase traincls to 80 
# CUDA_VISIBLE_DEVICES=4 python morjio_scripts/visual_info_transfer/trainer_866_1203.py --manualSeed 42 \
# --cls_weight 0.001 --cls_weight_unseen 0.001 --nclass_all 1204 --syn_num 250 --val_every 1 \
# --cuda --netG_name MLP_G --netD_name MLP_D \
# --nepoch 150 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
# --dataset lvis --batch_size 32 --nz 300 --attSize 512 --resSize 1024 --gan_epoch_budget 50000 \
# --lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
# --pretrain_classifier work_dirs/faster_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1_remove_rare_base_training/base_model_random_init_bbox_head.pth \
# --pretrain_classifier_unseen data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/pretrained_classifier/unseen_Classifier_textemb.pth \
# --class_embedding data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/866_337_fsd_split/class_embedding/lvis_all_classnames_text_embedding_use_prompts_7_original_order_with_bg_switch_bg_for_mmdet2.npy \
# --dataroot data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/866_337_fsd_split  \
# --testsplit lvis_unseen_test_0.6_0.3 \
# --trainsplit lvis_seen_train_0.6_0.3 \
# --classes_split 866_337 \
# --lz_ratio 0.01 \
# --outname checkpoints/fsd_866_1203 \
# --traincls_classifier base_few_zero_shot \
# --mixup



## bert word embedding, increase traincls to 80 
CUDA_VISIBLE_DEVICES=5 python morjio_scripts/visual_info_transfer/trainer_866_1203.py --manualSeed 42 \
--cls_weight 0.001 --cls_weight_unseen 0.001 --nclass_all 1204 --syn_num 250 --val_every 1 \
--cuda --netG_name MLP_G --netD_name MLP_D \
--nepoch 150 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--dataset lvis --batch_size 32 --nz 300 --attSize 768 --resSize 1024 --gan_epoch_budget 50000 \
--lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
--pretrain_classifier work_dirs/faster_rcnn_r101_fpn_sample1e-3_mstrain_1x_lvis_v1_remove_rare_base_training/base_model_random_init_bbox_head.pth \
--pretrain_classifier_unseen data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/pretrained_classifier/unseen_Classifier_textemb.pth \
--class_embedding data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/866_337_fsd_split/class_embedding/lvis_all_classnames_bert_word_embedding_with_bg_switch_bg_for_mmdet2.npy \
--dataroot data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/866_337_fsd_split  \
--testsplit lvis_unseen_test_0.6_0.3 \
--trainsplit lvis_seen_train_0.6_0.3 \
--classes_split 866_337 \
--lz_ratio 0.01 \
--outname checkpoints/fsd_866_1203 \
--traincls_classifier base_few_zero_shot 


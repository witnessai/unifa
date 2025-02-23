## CLIP text embedding, increase traincls to 80 
CUDA_VISIBLE_DEVICES=3 python morjio_scripts/visual_info_transfer/trainer_65_80.py --manualSeed 42 \
--cls_weight 0.001 --cls_weight_unseen 0.001 --nclass_all 81 --syn_num 250 --val_every 1 \
--cuda --netG_name MLP_G --netD_name MLP_D \
--nepoch 100 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--dataset coco --batch_size 32 --nz 300 --attSize 512 --resSize 1024 --gan_epoch_budget 38000 \
--lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
--pretrain_classifier work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth \
--pretrain_classifier_unseen data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/pretrained_classifier/unseen_Classifier_textemb.pth \
--class_embedding data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/class_embedding/coco_text_embedding_use_prompts_7_originalorder_with_bg_switch_bg_for_mmdet2.npy \
--dataroot data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/tfa/10shot  \
--testsplit test_0.6_0.3 \
--trainsplit train_0.6_0.3 \
--classes_split 65_8_7 \
--lz_ratio 0.01 \
--outname checkpoints/fsd_65_80 \
--traincls_classifier base_few_zero_shot \

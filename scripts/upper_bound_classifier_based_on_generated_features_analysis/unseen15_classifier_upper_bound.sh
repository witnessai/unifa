# cwp morjio_scripts/train_feature_generator/my_codes
CUDA_VISIBLE_DEVICES=1 python morjio_scripts/train_feature_generator/train_unseen15_classifier_on_gt_zeroshot_feature.py --manualSeed 42 \
--cls_weight 0.001 --cls_weight_zero_shot 0.001 --nclass_all 81 --syn_num 250 --val_every 1 \
--cuda --netG_name MLP_G --netD_name MLP_D \
--nepoch 100 --nepoch_cls 15 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
--dataset coco --batch_size 32 --nz 300 --attSize 300 --resSize 1024 --gan_epoch_budget 38000 \
--lr 0.00005 --lr_step 30 --lr_cls 0.0001 \
--pretrain_classifier work_dirs/asd_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth \
--pretrain_classifier_zero_shot data/coco/any_shot_detection/zero_shot_Classifier.pth \
--class_embedding data/coco/any_shot_detection/fasttext_switch_bg.npy \
--dataroot data/coco/any_shot_detection/base_few_shot_det/multi_labels/few_zero_shot \
--testsplit test_0.6_0.3 \
--trainsplit train_0.6_0.3 \
--classes_split 65_15 \
--lz_ratio 0.01 \
--outname checkpoints/asd_65_8_7/unseen15_classifier_upper_bound/train_2nd
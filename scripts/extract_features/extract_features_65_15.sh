# 基类检测器：训练视觉特征提取（基类特征）
# python morjio_scripts/extract_feats_using_pretrained_detector.py configs/detection/asd/coco/extract_visual_features.py --classes seen --load_from ./work_dirs/asd_r101_fpn_coco_base-training/iter_110000_trainvaldifferent_3gpus.pth --save_dir data/coco/any_shot_detection/base_det --data_split train

# data_split用于控制挑选数据集（训练视觉特征，还是测试视觉特征），classes用于控制挑选类别集

# 基类检测器：测试视觉特征提取（零样本类特征）
# python morjio_scripts/extract_feats_using_pretrained_detector.py configs/detection/asd/coco/extract_visual_features.py --classes unseen --load_from ./work_dirs/asd_r101_fpn_coco_base-training/iter_110000_trainvaldifferent_3gpus.pth --save_dir data/coco/any_shot_detection/base_det --data_split test


# 基类+小样本类检测器：训练视觉特征提取（基类特征）
# python morjio_scripts/extract_feats_using_pretrained_detector.py configs/detection/asd/coco/base_few_shot_det_extract_visual_features.py --classes seen --load_from ./work_dirs/rahman_ce_asd_65_8_7_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth --save_dir data/coco/any_shot_detection/base_few_shot_det --data_split train

# 基类+小样本类检测器：测试视觉特征提取（零样本类特征）
# python morjio_scripts/extract_feats_using_pretrained_detector.py configs/detection/asd/coco/base_few_shot_det_extract_visual_features.py --classes unseen --load_from ./work_dirs/rahman_ce_asd_65_8_7_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth --save_dir data/coco/any_shot_detection/base_few_shot_det --data_split test


# 基类+小样本类检测器：15类unseen测试特征的提取
# CUDA_VISIBLE_DEVICES=2 python morjio_scripts/extract_feats_using_pretrained_detector.py configs/detection/asd/coco/base_few_shot_det_extract_visual_features.py --classes unseen --load_from ./work_dirs/rahman_ce_asd_65_8_7_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth --save_dir data/coco/any_shot_detection/base_few_shot_det/multi_labels/base_few_zero_shot --data_split train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制






# ========================================  TFA  ========================================
# visual_info_transfer, 基类+小样本类检测器：训练视觉特征提取（基类+小样本类特征）,10shot
# CUDA_VISIBLE_DEVICES=5 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/base_few_shot_det_extract_feats_in_visual_info_transfer_trainset_base_few.py --classes unseen --load_from ./work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split --data_split train
# 之前加载的检测模型为./work_dirs/rahman_ce_asd_65_8_7_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth
#65_15_fsd_split_train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制

# visual_info_transfer, 基类+小样本类检测器：测试视觉特征提取（小样本类特征），10shot
# CUDA_VISIBLE_DEVICES=5 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/base_few_shot_det_extract_feats_in_visual_info_transfer_testset_few.py --classes unseen --load_from ./work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/testset --data_split train
# 之前加载的检测模型为./work_dirs/rahman_ce_asd_65_8_7_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth
#65_15_fsd_split_train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制

# visual_info_transfer, 基类+小样本类检测器：微调训练集视觉特征提取（基类+小样本类特征），10shot
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/base_few_shot_det_extract_feats_in_visual_info_transfer_finetuneset_base_few.py --classes unseen --load_from ./work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/finetuneset --data_split train
# 之前加载的检测模型为./work_dirs/rahman_ce_asd_65_8_7_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth
# 65_15_fsd_split_train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制


# visual_info_transfer, 基类+小样本类检测器：测试集视觉特征提取（基类+小样本类特征）
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/base_few_shot_det_extract_feats_in_visual_info_transfer_testset_base_few.py --classes unseen --load_from ./work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/testsetall --data_split train
# 之前加载的检测模型为./work_dirs/rahman_ce_asd_65_8_7_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth
# 65_15_fsd_split_train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制


# visual_info_transfer, 基类+小样本类检测器：微调训练集视觉特征提取（基类+小样本类特征），1shot
# CUDA_VISIBLE_DEVICES=1 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/base_few_shot_det_extract_feats_in_visual_info_transfer_1shot_finetuneset_base_few.py --classes unseen --load_from ./work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/1shot/finetuneset --data_split train
# 65_15_fsd_split_train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制


# visual_info_transfer, 基类+小样本类检测器：微调训练集视觉特征提取（基类+小样本类特征），2shot
# CUDA_VISIBLE_DEVICES=1 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/base_few_shot_det_extract_feats_in_visual_info_transfer_2shot_finetuneset_base_few.py --classes unseen --load_from ./work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/2shot/finetuneset --data_split train

# visual_info_transfer, 基类+小样本类检测器：微调训练集视觉特征提取（基类+小样本类特征），3shot
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/base_few_shot_det_extract_feats_in_visual_info_transfer_3shot_finetuneset_base_few.py --classes unseen --load_from ./work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/3shot/finetuneset --data_split train

# visual_info_transfer, 基类+小样本类检测器：微调训练集视觉特征提取（基类+小样本类特征），5shot
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/base_few_shot_det_extract_feats_in_visual_info_transfer_5shot_finetuneset_base_few.py --classes unseen --load_from ./work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/5shot/finetuneset --data_split train



# ========================================  FSCE  ========================================
# visual_info_transfer, 基类+小样本类检测器：微调训练集视觉特征提取（基类+小样本类特征），1shot
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/fsce_base_few_shot_det_extract_feats_in_visual_info_transfer_1shot_finetuneset_base_few.py  --load_from ./work_dirs/fsd_65_15_fsce_r101_fpn_coco_1shot-fine-tuning/iter_30000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/1shot/finetuneset --data_split train 
# --save_file_prefix 
# 65_15_fsd_split_train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制

# visual_info_transfer, 基类+小样本类检测器：训练集视觉特征提取（基类+小样本类特征），1shot
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/fsce_base_few_shot_det_extract_feats_in_visual_info_transfer_1shot_trainset_base_few.py  --load_from ./work_dirs/fsd_65_15_fsce_r101_fpn_coco_1shot-fine-tuning/iter_30000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/1shot/ --data_split train 
# --save_file_prefix 
# 65_15_fsd_split_train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制

# visual_info_transfer, 基类+小样本类检测器：测试视觉特征提取（小样本类特征），1shot特征提取器
# CUDA_VISIBLE_DEVICES=1 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/fsce_base_few_shot_det_extract_feats_in_visual_info_transfer_1shot_testset_few.py  --load_from ./work_dirs/fsd_65_15_fsce_r101_fpn_coco_1shot-fine-tuning/iter_30000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/1shot/ --save_split test
# --save_file_prefix 
# 65_15_fsd_split_train

# visual_info_transfer, 基类+小样本类检测器：微调训练集视觉特征提取（基类+小样本类特征），2shot
# CUDA_VISIBLE_DEVICES=5 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/fsce_base_few_shot_det_extract_feats_in_visual_info_transfer_2shot_finetuneset_base_few.py  --load_from ./work_dirs/fsd_65_15_fsce_r101_fpn_coco_2shot-fine-tuning/iter_30000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/2shot/finetuneset --data_split train 
# --save_file_prefix 
# 65_15_fsd_split_train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制

# visual_info_transfer, 基类+小样本类检测器：训练集视觉特征提取（基类+小样本类特征），2shot
# CUDA_VISIBLE_DEVICES=5 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/fsce_base_few_shot_det_extract_feats_in_visual_info_transfer_2shot_trainset_base_few.py  --load_from ./work_dirs/fsd_65_15_fsce_r101_fpn_coco_2shot-fine-tuning/iter_30000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/2shot/ --data_split train 
# --save_file_prefix 
# 65_15_fsd_split_train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制

# visual_info_transfer, 基类+小样本类检测器：测试视觉特征提取（小样本类特征），2shot特征提取器
# CUDA_VISIBLE_DEVICES=3 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/fsce_base_few_shot_det_extract_feats_in_visual_info_transfer_2shot_testset_few.py  --load_from ./work_dirs/fsd_65_15_fsce_r101_fpn_coco_2shot-fine-tuning/iter_30000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/2shot/ --save_split test 
# --save_file_prefix 
# 65_15_fsd_split_train


# visual_info_transfer, 基类+小样本类检测器：微调训练集视觉特征提取（基类+小样本类特征），3shot
# CUDA_VISIBLE_DEVICES=4 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/fsce_base_few_shot_det_extract_feats_in_visual_info_transfer_3shot_finetuneset_base_few.py  --load_from ./work_dirs/fsd_65_15_fsce_r101_fpn_coco_3shot-fine-tuning/iter_30000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/3shot/finetuneset --data_split train 
# --save_file_prefix 
# 65_15_fsd_split_train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制

# visual_info_transfer, 基类+小样本类检测器：训练集视觉特征提取（基类+小样本类特征），3shot
# CUDA_VISIBLE_DEVICES=4 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/fsce_base_few_shot_det_extract_feats_in_visual_info_transfer_3shot_trainset_base_few.py  --load_from ./work_dirs/fsd_65_15_fsce_r101_fpn_coco_3shot-fine-tuning/iter_30000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/3shot/ --data_split train 
# --save_file_prefix 
# 65_15_fsd_split_train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制

# visual_info_transfer, 基类+小样本类检测器：测试视觉特征提取（小样本类特征），3shot特征提取器
# CUDA_VISIBLE_DEVICES=5 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/fsce_base_few_shot_det_extract_feats_in_visual_info_transfer_3shot_testset_few.py  --load_from ./work_dirs/fsd_65_15_fsce_r101_fpn_coco_3shot-fine-tuning/iter_30000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/3shot/ --save_split test 
# --save_file_prefix 
# 65_15_fsd_split_train


# visual_info_transfer, 基类+小样本类检测器：微调训练集视觉特征提取（基类+小样本类特征），5shot
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/fsce_base_few_shot_det_extract_feats_in_visual_info_transfer_5shot_finetuneset_base_few.py  --load_from ./work_dirs/fsd_65_15_fsce_r101_fpn_coco_5shot-fine-tuning/iter_30000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/5shot/finetuneset --data_split train 
# --save_file_prefix 
# 65_15_fsd_split_train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制

# visual_info_transfer, 基类+小样本类检测器：训练集视觉特征提取（基类+小样本类特征），5shot
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/fsce_base_few_shot_det_extract_feats_in_visual_info_transfer_5shot_trainset_base_few.py  --load_from ./work_dirs/fsd_65_15_fsce_r101_fpn_coco_5shot-fine-tuning/iter_30000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/5shot/ --data_split train 
# --save_file_prefix 
# 65_15_fsd_split_train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制

# visual_info_transfer, 基类+小样本类检测器：测试视觉特征提取（小样本类特征），5shot特征提取器
# CUDA_VISIBLE_DEVICES=3 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/fsce_base_few_shot_det_extract_feats_in_visual_info_transfer_5shot_testset_few.py  --load_from ./work_dirs/fsd_65_15_fsce_r101_fpn_coco_5shot-fine-tuning/iter_30000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/5shot/ --save_split test 
# --save_file_prefix 
# 65_15_fsd_split_train


# visual_info_transfer, 基类+小样本类检测器：微调训练集视觉特征提取（基类+小样本类特征），10shot
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/fsce_base_few_shot_det_extract_feats_in_visual_info_transfer_10shot_finetuneset_base_few.py  --load_from ./work_dirs/fsd_65_15_fsce_r101_fpn_coco_10shot-fine-tuning/iter_30000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/10shot/finetuneset --data_split train 
# --save_file_prefix 
# 65_15_fsd_split_train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制

# visual_info_transfer, 基类+小样本类检测器：训练集视觉特征提取（基类+小样本类特征），10shot
CUDA_VISIBLE_DEVICES=0 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/fsce_base_few_shot_det_extract_feats_in_visual_info_transfer_10shot_trainset_base_few.py  --load_from ./work_dirs/fsd_65_15_fsce_r101_fpn_coco_10shot-fine-tuning/iter_30000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/10shot/ --data_split train 
# --save_file_prefix 
# 65_15_fsd_split_train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制

# visual_info_transfer, 基类+小样本类检测器：测试视觉特征提取（小样本类特征），10shot特征提取器
# CUDA_VISIBLE_DEVICES=4 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/fsce_base_few_shot_det_extract_feats_in_visual_info_transfer_10shot_testset_few.py  --load_from ./work_dirs/fsd_65_15_fsce_r101_fpn_coco_10shot-fine-tuning/iter_30000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/fsce/10shot/ --save_split test 
# --save_file_prefix 
# 65_15_fsd_split_train

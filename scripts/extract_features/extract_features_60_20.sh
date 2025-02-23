# visual_info_transfer, 基类+小样本类检测器：训练视觉特征提取（基类+小样本类特征）,30shot, fsce, 提取特征完要rename为train_0.6_0.3_*_base_training.npy
# CUDA_VISIBLE_DEVICES=3 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/fsce/extract_feats_coco/fsce_r101_fpn_coco_30shot-fine-tuning_extract_feats_in_visual_info_transfer_base_training.py --classes unseen --load_from ./work_dirs/fsce_r101_fpn_coco_30shot-fine-tuning_34server/iter_40000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/30shot --data_split train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制

# visual_info_transfer, 基类+小样本类检测器：微调训练集视觉特征提取（基类+小样本类特征），30shot, fsce， 提取特征完要rename为train_0.6_0.3_*_fine_tuning.npy
# CUDA_VISIBLE_DEVICES=3 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/fsce/extract_feats_coco/fsce_r101_fpn_coco_30shot-fine-tuning_extract_feats_in_visual_info_transfer_fine_tuning.py --classes unseen --load_from ./work_dirs/fsce_r101_fpn_coco_30shot-fine-tuning_34server/iter_40000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/30shot --data_split train


# visual_info_transfer, 基类+小样本类检测器：测试视觉特征提取（小样本类特征），30shot, fsce，提取特征完要rename为test_0.6_0.3_*.npy
CUDA_VISIBLE_DEVICES=0 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/fsce/extract_feats_coco/fsce_r101_fpn_coco_30shot-fine-tuning_extract_feats_in_visual_info_transfer_test.py --classes unseen --load_from ./work_dirs/fsce_r101_fpn_coco_30shot-fine-tuning_34server/iter_40000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/30shot --data_split train


# visual_info_transfer, 基类+小样本类检测器：训练视觉特征提取（基类+小样本类特征）,10shot, fsce, 提取特征完要rename为train_0.6_0.3_*_base_training.npy
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/fsce/extract_feats_coco/fsce_r101_fpn_coco_10shot-fine-tuning_extract_feats_in_visual_info_transfer_base_training.py --classes unseen --load_from ./work_dirs/fsce_r101_fpn_coco_10shot-fine-tuning_34server/iter_30000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/10shot --data_split train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制

# visual_info_transfer, 基类+小样本类检测器：微调训练集视觉特征提取（基类+小样本类特征），10shot, fsce， 提取特征完要rename为train_0.6_0.3_*_fine_tuning.npy
# CUDA_VISIBLE_DEVICES=2 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/fsce/extract_feats_coco/fsce_r101_fpn_coco_10shot-fine-tuning_extract_feats_in_visual_info_transfer_fine_tuning.py --classes unseen --load_from ./work_dirs/fsce_r101_fpn_coco_10shot-fine-tuning_34server/iter_30000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/10shot --data_split train

# visual_info_transfer, 基类+小样本类检测器：测试视觉特征提取（小样本类特征），10shot, fsce，提取特征完要rename为test_0.6_0.3_*.npy
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/fsce/extract_feats_coco/fsce_r101_fpn_coco_10shot-fine-tuning_extract_feats_in_visual_info_transfer_test.py --classes unseen --load_from ./work_dirs/fsce_r101_fpn_coco_10shot-fine-tuning_34server/iter_30000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/fsce/10shot --data_split train


# visual_info_transfer, 基类+小样本类检测器：训练视觉特征提取（基类+小样本类特征）,30shot, tfa, 提取特征完要rename为train_0.6_0.3_*_base_training.npy
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/tfa/extract_feats_coco/tfa_r101_fpn_coco_30shot-fine-tuning_extract_feats_in_visual_info_transfer_base_training.py --classes unseen --load_from ./work_dirs/tfa_r101_fpn_coco_30shot-fine-tuning_34server/iter_240000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/30shot --data_split train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制

# visual_info_transfer, 基类+小样本类检测器：微调训练集视觉特征提取（基类+小样本类特征），30shot, tfa， 提取特征完要rename为train_0.6_0.3_*_fine_tuning.npy
# CUDA_VISIBLE_DEVICES=2 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/tfa/extract_feats_coco/tfa_r101_fpn_coco_30shot-fine-tuning_extract_feats_in_visual_info_transfer_fine_tuning.py --classes unseen --load_from ./work_dirs/tfa_r101_fpn_coco_30shot-fine-tuning_34server/iter_240000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/30shot --data_split train

# visual_info_transfer, 基类+小样本类检测器：测试视觉特征提取（小样本类特征），30shot, tfa，提取特征完要rename为test_0.6_0.3_*.npy
# CUDA_VISIBLE_DEVICES=2 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/tfa/extract_feats_coco/tfa_r101_fpn_coco_30shot-fine-tuning_extract_feats_in_visual_info_transfer_test.py --classes unseen --load_from ./work_dirs/tfa_r101_fpn_coco_30shot-fine-tuning_34server/iter_240000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/30shot --data_split train



# visual_info_transfer, 基类+小样本类检测器：训练视觉特征提取（基类+小样本类特征）,10shot, tfa, 提取特征完要rename为train_0.6_0.3_*_base_training.npy
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/tfa/extract_feats_coco/tfa_r101_fpn_coco_10shot-fine-tuning_extract_feats_in_visual_info_transfer_base_training.py --classes unseen --load_from ./work_dirs/tfa_r101_fpn_coco_10shot-fine-tuning_34server/iter_160000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/10shot --data_split train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制

# visual_info_transfer, 基类+小样本类检测器：微调训练集视觉特征提取（基类+小样本类特征），10shot, tfa， 提取特征完要rename为train_0.6_0.3_*_fine_tuning.npy
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/tfa/extract_feats_coco/tfa_r101_fpn_coco_10shot-fine-tuning_extract_feats_in_visual_info_transfer_fine_tuning.py --classes unseen --load_from ./work_dirs/tfa_r101_fpn_coco_10shot-fine-tuning_34server/iter_160000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/10shot --data_split train

# visual_info_transfer, 基类+小样本类检测器：测试视觉特征提取（小样本类特征），10shot, tfa，提取特征完要rename为test_0.6_0.3_*.npy
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/tfa/extract_feats_coco/tfa_r101_fpn_coco_10shot-fine-tuning_extract_feats_in_visual_info_transfer_test.py --classes unseen --load_from ./work_dirs/tfa_r101_fpn_coco_10shot-fine-tuning_34server/iter_160000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/60_20_fsd_split/tfa/10shot --data_split train

# visual_info_transfer, 基类+小样本类检测器：训练集视觉特征提取（基类+小样本类特征），10shot
# CUDA_VISIBLE_DEVICES=0 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/tfa_base_det_extract_feats_in_visual_info_transfer_48_17_trainset_base.py  --load_from ./work_dirs/fsd_48_17_r101_fpn_coco_base-training/iter_110000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/48_17_fsd_split/tfa/0shot/ --data_split train 
# --save_file_prefix 
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制




# visual_info_transfer, 基类+小样本类检测器：测试视觉特征提取（小样本类特征），10shot特征提取器
CUDA_VISIBLE_DEVICES=1 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/tfa_base_det_extract_feats_in_visual_info_transfer_48_17_testset_novel.py  --load_from ./work_dirs/fsd_48_17_r101_fpn_coco_base-training/iter_110000.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/48_17_fsd_split/tfa/0shot/ --save_split test 
# --save_file_prefix 
# 65_15_fsd_split_train
## visual_info_transfer, 基类+小样本类检测器：测试视觉特征提取（小样本类特征），10shot
CUDA_VISIBLE_DEVICES=4 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/asd/coco/base_few_shot_det_extract_feats_in_visual_info_transfer_10shot_testset_few_73_7.py --classes unseen --load_from ./work_dirs/fsd_65_15_r101_fpn_coco_base-training/base_model_random_init_bbox_head.pth --save_dir data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/73_7_fsd_split/tfa/10shot --data_split train
# 之前加载的检测模型为./work_dirs/rahman_ce_asd_65_8_7_r101_fpn_coco_10shot-fine-tuning/iter_160000.pth
#65_15_fsd_split_train
# classes该参数无影响，提取什么类别，由config文件中classes（ALL_CLASSES, BASE_CLASSES等）控制
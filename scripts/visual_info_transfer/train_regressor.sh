## use text embedding
# CUDA_VISIBLE_DEVICES=2 python morjio_scripts/visual_info_transfer/train_regressor.py


## use word embedding
# CUDA_VISIBLE_DEVICES=1 python morjio_scripts/visual_info_transfer/train_regressor.py --save_path data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/regressor_wordemb_tfa.pth --class_embedding data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/fasttext_switch_bg.npy


## use qwen2 embedding
CUDA_VISIBLE_DEVICES=0 python morjio_scripts/visual_info_transfer/train_regressor.py --save_path data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/regressor_qwenemb_tfa.pth --class_embedding /home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/coco/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/65_15_fsd_split/class_embedding/qwen2.5_embedding_65_15_split_originalorder_switch_bg_for_mmdet2.npy
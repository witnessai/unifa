## fsce+CPE loss, split1, 3shot, fine-tuning
# CUDA_VISIBLE_DEVICES=4 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_5shot-fine-tuning.py  --load_from ./work_dirs/fsce_r101_fpn_contrastive-loss_voc-split1_5shot-fine-tuning/iter_9000.pth --save_dir data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce_add_cpe/5shot/ --data_split train


## fsce+CPE loss, split1, 3shot, fine-tuning
# CUDA_VISIBLE_DEVICES=4 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_3shot-fine-tuning.py  --load_from ./work_dirs/fsce_r101_fpn_contrastive-loss_voc-split1_3shot-fine-tuning/iter_8000.pth --save_dir data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce_add_cpe/3shot/ --data_split train

## fsce+CPE loss, split1, 3shot, test
# CUDA_VISIBLE_DEVICES=3 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/fsce/voc/split1/fsce_r101_fpn_contrastive-loss_voc-split1_3shot-fine-tuning_extract_feats.py  --load_from ./work_dirs/fsce_r101_fpn_contrastive-loss_voc-split1_3shot-fine-tuning/iter_8000.pth --save_dir data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce_add_cpe/3shot/ --data_split test


## fsce, split1, 5shot, fine-tuning
# CUDA_VISIBLE_DEVICES=5 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_5shot-fine-tuning.py  --load_from ./work_dirs/fsce_r101_fpn_voc-split1_5shot-fine-tuning/iter_9000.pth --save_dir data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce/5shot/ --data_split train


## fsce, split1, 3shot, base training
# CUDA_VISIBLE_DEVICES=3 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_base-training_extract_feats.py  --load_from ./work_dirs/fsce_r101_fpn_voc-split1_base-training/iter_18000.pth --save_dir data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce/3shot/ --data_split train

## fsce, split1, 3shot, fine-tuning
# CUDA_VISIBLE_DEVICES=3 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_3shot-fine-tuning.py  --load_from ./work_dirs/fsce_r101_fpn_voc-split1_3shot-fine-tuning/iter_8000.pth --save_dir data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce/3shot/ --data_split train

## fsce, split1, 3shot, test
# CUDA_VISIBLE_DEVICES=3 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/fsce/voc/split1/fsce_r101_fpn_voc-split1_3shot-fine-tuning_extract_feats.py  --load_from ./work_dirs/fsce_r101_fpn_voc-split1_3shot-fine-tuning/iter_8000.pth --save_dir data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/fsce/3shot/ --data_split test

## tfa, split1, 5shot, fine-tuning
# CUDA_VISIBLE_DEVICES=2 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_5shot-fine-tuning.py  --load_from ./work_dirs/tfa_r101_fpn_voc-split1_5shot-fine-tuning/iter_20000.pth --save_dir data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/5shot/ --data_split train

## tfa, split1, 3shot, base training
# CUDA_VISIBLE_DEVICES=1 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_base-training_extract_feats.py  --load_from ./work_dirs/tfa_r101_fpn_voc-split1_base-training/iter_18000.pth --save_dir data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/3shot/ --data_split train

## tfa, split1, 3shot, fine-tuning
# CUDA_VISIBLE_DEVICES=2 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_3shot-fine-tuning.py  --load_from ./work_dirs/tfa_r101_fpn_voc-split1_3shot-fine-tuning/iter_12000.pth --save_dir data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/3shot/ --data_split train

## tfa, split1, 3shot, test
# CUDA_VISIBLE_DEVICES=3 python morjio_scripts/extract_feats_in_visual_info_transfer.py configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_3shot-fine-tuning_extract_feats.py  --load_from ./work_dirs/tfa_r101_fpn_voc-split1_3shot-fine-tuning/iter_12000.pth --save_dir data/voc/any_shot_detection_extracted_feats_and_embedding_classifier/visual_info_transfer/15_5_fsd_split/tfa/3shot/ --data_split test
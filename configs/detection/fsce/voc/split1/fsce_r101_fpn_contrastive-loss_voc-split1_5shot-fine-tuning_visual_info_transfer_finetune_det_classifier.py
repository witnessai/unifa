_base_ = [
    '../../../_base_/datasets/fine_tune_based/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py',
    '../../fsce_r101_fpn_contrastive_loss.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        type='FewShotVOCDefaultDataset',
        ann_cfg=[dict(method='FSCE', setting='SPLIT1_5SHOT')],
        num_novel_shots=5,
        num_base_shots=5,
        classes='ALL_CLASSES_SPLIT1'),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=4500,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=4500)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=200, gamma=0.5, step=[5000, 7000])
runner = dict(max_iters=9000)
custom_hooks = [
    dict(
        type='ContrastiveLossDecayHook',
        decay_steps=(4000, 7000),
        decay_rate=0.5)
]
model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='ContrastiveBBoxHead_visual_info_transfer_voc_15_5_5shot_finetune_det_classifier_fsce_add_cpe',
            with_weight_decay=True,
            loss_contrast=dict(iou_threshold=0.7, loss_weight=0.5))))
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/fsce/README.md for more details.
load_from = ('work_dirs/fsce_r101_fpn_voc-split1_base-training/'
             'base_model_random_init_bbox_head.pth')

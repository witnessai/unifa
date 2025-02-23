_base_ = [
    '../../../_base_/datasets/fine_tune_based/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py', '../../fsce_r101_fpn.py',
    '../../../_base_/default_runtime.py'
]
model = dict(
    type='FSCE',
    frozen_parameters=['backbone'],
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101, frozen_stages=4),
    roi_head=dict(
        bbox_head=dict(
            type='CosineSimBBoxHead_visual_info_transfer_voc_15_5_3shot_finetune_det_classifier_fsce',
            num_shared_fcs=2,
            num_classes=20,
            scale=20)),
    train_cfg=dict(
        rpn_proposal=dict(max_per_img=2000),
        rcnn=dict(
            assigner=dict(pos_iou_thr=0.4, neg_iou_thr=0.4, min_pos_iou=0.4),
            sampler=dict(num=256))))
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        type='FewShotVOCDefaultDataset',
        ann_cfg=[dict(method='FSCE', setting='SPLIT1_3SHOT')],
        num_novel_shots=3,
        num_base_shots=3,
        classes='ALL_CLASSES_SPLIT1'),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=4000,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=4000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=200, gamma=0.5, step=[4000, 6000])
runner = dict(max_iters=8000)
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/fsce/README.md for more details.
load_from = ('work_dirs/fsce_r101_fpn_voc-split1_base-training/'
             'base_model_random_init_bbox_head.pth')

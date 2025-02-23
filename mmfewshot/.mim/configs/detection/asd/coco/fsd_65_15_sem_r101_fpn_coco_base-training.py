_base_ = [
    '../../_base_/datasets/any_shot_detection/base_coco.py',
    '../../_base_/schedules/schedule.py',
    '../../_base_/models/faster_rcnn_r50_caffe_fpn.py',
    '../../_base_/default_runtime.py'
]
lr_config = dict(warmup_iters=1000, step=[85000, 100000])
runner = dict(max_iters=110000)
# model settings
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(
        bbox_head=dict(
            type='Shared2FCSemanticBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=65,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            semantic_dims=300,
            base_class=False,
            reg_with_semantic=False,
            share_semantic=False,
            # voc_path='data/coco/vocabulary_w2v.txt',
            vec_path='data/coco/word_w2v_with_learnable_bg_65_15.txt',
            # vec_path='data/coco/word_w2v_withbg_65_15.txt',
            reg_class_agnostic=True,
            loss_semantic=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            
        )
    
    )

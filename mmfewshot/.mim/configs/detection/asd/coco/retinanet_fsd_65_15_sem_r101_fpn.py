# refer to tfa_r101_fpn.py
_base_ = ['/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/configs/detection/_base_/models/retinanet_r50_fpn.py']
model = dict(
    type='RetinaNet_fsd',
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    bbox_head=dict(
        type='RetinaSemHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        with_semantic=True,
        semantic_dims=300,
        reg_with_semantic=False,
        share_semantic=False,
        vec_path='data/coco/word_w2v_with_learnable_bg_65_15.txt',
        voc_path='data/coco/vocabulary_w2v.txt',
        ),
)
            


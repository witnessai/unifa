_base_ = [
    '../../_base_/datasets/any_shot_detection/base_coco_48_17.py',
    '../../_base_/schedules/schedule.py', '../../_base_/models/faster_rcnn_r50_caffe_fpn.py',
    '../../_base_/default_runtime.py'
]

evaluation = dict(interval=8000)
checkpoint_config = dict(interval=8000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=10, step=[14400])
runner = dict(max_iters=16000)
model = dict(
    type='TFA',
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(
        type='StandardRoIHead_visual_info_transfer_48_17_two_softmax_from_suzsd_48_65_qwen',
        bbox_head=dict(
            type='Shared2FCBBoxHead_visual_info_transfer_coco_48_17_0shot_two_softmax_from_suzsd_48_65_qwen',
            # fsd=True,
            # gfsd=False,
            fsd=False,
            gfsd=True,
            num_classes=65
            )),
)
load_from = ('work_dirs/fsd_48_17_r101_fpn_coco_base-training/'
             'base_model_random_init_bbox_head.pth')



# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 672), (1333, 704), (1333, 736),
                   (1333, 768), (1333, 800)],
        keep_ratio=True,
        multiscale_mode='value'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# classes splits are predefined in FewShotCocoDataset
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='FewShotCocoDataset_48_17',
        save_dataset=False,
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/coco/annotations/instances_train2014_seen_48_17.json')
        ],
        img_prefix=data_root,
        pipeline=train_pipeline,
        classes='BASE_CLASSES'),
    val=dict(
        type='FewShotCocoDataset_48_17',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/coco/annotations/instances_val2017.json')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes='BASE_CLASSES'),
    test=dict(
        type='FewShotCocoDataset_48_17',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/coco/annotations/instances_val2014_gzsd_48_17.json')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        classes='ALL_CLASSES'))
evaluation = dict(
    interval=4000,
    metric='bbox',
    classwise=True,
    # class_splits=['BASE_CLASSES', 'NOVEL_CLASSES']
    class_splits=['BASE_CLASSES', 'NOVEL_CLASSES']
    )



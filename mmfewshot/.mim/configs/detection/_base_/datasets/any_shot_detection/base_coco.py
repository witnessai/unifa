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
        type='AnyShotCocoDataset',
        save_dataset=False,
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/any_shot_ann/coco/annotations/instances_train2014_base_65_15.json')
        ],
        img_prefix=data_root,
        pipeline=train_pipeline,
        classes='BASE_CLASSES'),
    val=dict(
        type='AnyShotCocoDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/any_shot_ann/coco/annotations/instances_val2014_base_65_15.json')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes='BASE_CLASSES'),
    test=dict(
        type='AnyShotCocoDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/any_shot_ann/coco/annotations/instances_val2014_base_65_15.json')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        classes='BASE_CLASSES'))
evaluation = dict(interval=5000, metric='bbox', classwise=True)

_base_ = [
    '../../../_base_/datasets/fine_tune_based/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py', '../../tfa_r101_fpn.py',
    '../../../_base_/default_runtime.py'
]
# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 512), (1333, 544), (1333, 576),
                   (1333, 608), (1333, 640), (1333, 672), (1333, 704),
                   (1333, 736), (1333, 768), (1333, 800)],
        keep_ratio=True,
        multiscale_mode='value'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', with_bbox=True),  
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
            # dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ])
]
# classes splits are predefined in FewShotVOCDataset
data_root = 'data/VOCdevkit/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='FewShotVOCDefaultDataset',
        save_dataset=True,
        ann_cfg=[dict(method='TFA', setting='SPLIT1_3SHOT')],
        img_prefix=data_root,
        num_novel_shots=3,
        num_base_shots=3,
        pipeline=train_pipeline,
        classes='ALL_CLASSES_SPLIT1',
        use_difficult=False,
        instance_wise=False),
    val=dict(
        type='FewShotVOCDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes='ALL_CLASSES_SPLIT1'),
    test=dict(
        type='FewShotVOCDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        # test_mode=True,
        test_mode=False,
        classes='NOVEL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=12000,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=12000)
optimizer = dict(lr=0.001)
lr_config = dict(
    warmup_iters=10, step=[
        11000,
    ])
runner = dict(max_iters=12000)
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.
load_from = ('work_dirs/tfa_r101_fpn_voc-split1_base-training/'
             'base_model_random_init_bbox_head.pth')

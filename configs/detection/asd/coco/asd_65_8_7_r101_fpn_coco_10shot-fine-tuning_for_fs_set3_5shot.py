_base_ = [
    '../../_base_/datasets/any_shot_detection/any_shot_coco_65_8_7.py',
    '../../_base_/schedules/schedule.py', './fsd_65_15_r101_fpn.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility
# data = dict(
#     train=dict(
#         type='FewShotCocoDataset_65_15',
#         ann_cfg=[dict(method='TFA', setting='10SHOT')],
#         num_novel_shots=10,
#         num_base_shots=10))
# data = dict(
#     train=dict(
#             save_dataset=True,
#             type='AnyShotCocoDataset',
#             num_novel_shots=10,
#             num_base_shots=10,
#             classes='ALL_CLASSES',
#             instance_wise=False)
#         )
evaluation = dict(interval=5000)
checkpoint_config = dict(interval=5000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=10, step=[144000])
runner = dict(max_iters=160000)
model = dict(
    frozen_parameters=[
        'backbone', 'neck', 'rpn_head', 'roi_head.bbox_head.shared_fcs'
    ],
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(
        bbox_head=dict(
            num_classes=73, 
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),)))

# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.
# load_from = ('work_dirs/asd_r101_fpn_coco_base-training/'
#              'base_model_random_init_bbox_head.pth')

load_from = ('work_dirs/asd_65_8_7_r101_fpn_coco_base-training/'
             'base_model_random_init_bbox_head_for_fs_set3.pth')

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
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
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
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ])
]
# classes splits are predefined in FewShotCocoDataset
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        save_dataset=True,
        type='AnyShotCocoDataset_65_8_7',
        ann_cfg=[
            dict(
                type='ann_file',
                # ann_file='data/any_shot_ann_65_8_7_split/rahman_asd_full_box_10shot_trainval_42.json'
                # ann_file='data/any_shot_ann_65_8_7_split/rahman_asd_full_box_10shot_trainval_42_v3.json'
                ann_file='data/any_shot_ann_65_8_7_split/rahman_asd_full_box_5shot_trainval_42_v4_for_fs_set3.json'
                )
        ],
        img_prefix=data_root,
        num_novel_shots=None,
        num_base_shots=None,
        pipeline=train_pipeline,
        classes='BASE_FEW_SHOT_3_CLASSES',
        instance_wise=False),
    val=dict(
        type='AnyShotCocoDataset_65_8_7',
        ann_cfg=[
            dict(
                type='ann_file',
                # ann_file='data/few_shot_ann/coco/annotations/val.json'
                ann_file='data/coco/annotations/instances_val2014.json'
                )
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes='BASE_FEW_SHOT_3_CLASSES'
        ),
    test=dict(
        type='AnyShotCocoDataset_65_8_7',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/coco/annotations/instances_val2014.json'
                # ann_file='data/few_shot_ann/coco/annotations/val.json'
                # ann_file='data/coco/annotations/instances_shafin_test_morjio_for_asd.json'
                # ann_file='data/coco/annotations/instances_val2017.json'
                # ann_file='data/any_shot_ann/coco/annotations/finetune_dataset_seed42/full_box_10shot_trainval.json'
                )
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        # test_mode=True, # 加载80类
        test_mode=False, # 加载73类
        classes='BASE_FEW_SHOT_3_CLASSES'
        ))
evaluation = dict(
    interval=4000,
    metric='bbox',
    classwise=True,
    # class_splits=['BASE_CLASSES', 'NOVEL_CLASSES']
    class_splits=['BASE_CLASSES', 'FEW_SHOT_CLASSES_3', 'ZERO_SHOT_CLASSES_3']
    )

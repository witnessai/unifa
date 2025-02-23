_base_ = [
    '../../_base_/datasets/any_shot_detection/few_shot_coco_65_15.py',
    '../../_base_/schedules/schedule.py', './fsd_65_15_fsce_r101_fpn.py',
    '../../_base_/default_runtime.py'
]





# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.
# load_from = ('work_dirs/asd_r101_fpn_coco_base-training/'
#              'base_model_random_init_bbox_head.pth')

load_from = ('work_dirs/fsd_65_15_fsce_r101_fpn_coco_3shot-fine-tuning/'
             'iter_30000.pth')


evaluation = dict(interval=8000)
checkpoint_config = dict(interval=8000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=10, step=[14400])
runner = dict(max_iters=16000)

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='CosineSimBBoxHead_visual_info_transfer_coco_65_15_3shot_finetune_det_classifier'
            )),
)


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
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        save_dataset=True,
        type='FewShotCocoDataset_65_15',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/few_shot_ann_65_15_split/rahman_fsd_full_box_3shot_trainval.json'
                
                )
        ],
        img_prefix=data_root,
        num_novel_shots=None,
        num_base_shots=None,
        pipeline=train_pipeline,
        classes='ALL_CLASSES',
        instance_wise=False),
    val=dict(
        type='FewShotCocoDataset_65_15',
        ann_cfg=[
            dict(
                type='ann_file',
                # ann_file='data/few_shot_ann/coco/annotations/val.json'
                ann_file='data/coco/annotations/instances_val2014.json'
                )
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes='ALL_CLASSES'),
    test=dict(
        type='FewShotCocoDataset_65_15',
        ann_cfg=[
            dict(
                type='ann_file',
                # ann_file='data/coco/annotations/instances_val2014.json'
                # ann_file='data/few_shot_ann/coco/annotations/val.json'
                ann_file='data/coco/annotations/instances_shafin_test_morjio_for_asd.json'
                # ann_file='data/coco/annotations/instances_val2017.json'
                # ann_file='data/any_shot_ann/coco/annotations/finetune_dataset_seed42/full_box_10shot_trainval.json'
                )
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        classes='ALL_CLASSES'))
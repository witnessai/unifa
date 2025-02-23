_base_ = [
    '../../_base_/datasets/any_shot_detection/base_few_shot_coco_65_15_in_visual_info_transfer.py',
    '../../_base_/schedules/schedule.py', './fsd_65_15_r101_fpn.py',
    '../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotCocoDataset
# FewShotCocoDefaultDataset predefine ann_cfg for model reproducibility

evaluation = dict(interval=80000)
checkpoint_config = dict(interval=80000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=10, step=[144000])
runner = dict(max_iters=160000)
model = dict(
    type='TFA',
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(
        type='StandardRoIHead2',
        bbox_head=dict(
            type='Shared2FCBBoxHead2',
            num_classes=80, # some classes have no training process
        )), 
    test_cfg=dict(
        rcnn=dict(
            asd=False,
            gasd=False,
            dataset_name='coco',
            split='65_8_7'
        )
    )
)



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
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(1333, 800),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img'])
#         ])
# ]
# classes splits are predefined in FewShotCocoDataset
data_root = 'data/coco/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='FewShotCocoDataset_65_15',
        save_dataset=False,
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/few_shot_ann_65_15_split/rahman_fsd_full_box_3shot_trainval_42.json'
                
                )
        ],
        img_prefix=data_root,
        pipeline=train_pipeline,
        classes='ALL_CLASSES'
        # type='FewShotCocoDataset_65_15',
        # save_dataset=False,
        # ann_cfg=[
        #     dict(
        #         type='ann_file',
        #         ann_file='data/coco/annotations/instances_val2014.json')
        # ],
        # img_prefix=data_root,
        # pipeline=train_pipeline,
        # classes='NOVEL_CLASSES'
        ),
    val=dict(
        type='FewShotCocoDataset_65_15',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/coco/annotations/combine_base_few_shot_json_in_visual_info_transfer/base_few_shot_json_for_extract_feats_in_visual_info_transfer.json')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes='ALL_CLASSES'),
    test=dict(
        type='FewShotCocoDataset_65_15',
        save_dataset=False,
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/coco/annotations/instances_val2014.json')
        ],
        img_prefix=data_root,
        pipeline=train_pipeline,
        # test_mode=True,
        classes='NOVEL_CLASSES'))
evaluation = dict(interval=5000, metric='bbox', classwise=True)

_base_ = [
    '../../_base_/datasets/any_shot_detection/any_shot_coco_65_8_7.py',
    '../../_base_/schedules/schedule.py', './fsd_65_15_r101_fpn.py',
    '../../_base_/default_runtime.py'
]

evaluation = dict(interval=8000)
checkpoint_config = dict(interval=8000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=10, step=[14400])
runner = dict(max_iters=16000)
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(
        type='StandardRoIHead_visual_info_transfer_73_15_two_softmax_from_suzsd',
        bbox_head=dict(
            type='Shared2FCBBoxHead_visual_info_transfer_coco_73_15_10shot_two_softmax_from_suzsd',
            asd=True,
            gasd=False,
            # fsd=True,
            # gfsd=False,
            num_classes=80
            )),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            # gfsd=True,
            # fsd=False, 
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100, 
            split='65_8_7', 
            fs_set=3,
            dataset_name='coco'
            ))
    
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
    workers_per_gpu=8,
    train=dict(
        save_dataset=True,
        type='AnyShotCocoDataset_65_8_7',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/few_shot_ann_65_15_split/rahman_fsd_full_box_1shot_trainval.json'
                
                )
        ],
        img_prefix=data_root,
        num_novel_shots=None,
        num_base_shots=None,
        pipeline=train_pipeline,
        classes='ALL_CLASSES',
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
        classes='ALL_CLASSES'),
    test=dict(
        type='AnyShotCocoDataset_65_8_7',
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

# class type in AnyShotCocoDataset_65_8_7:
# ALL_CLASSES
# BASE_CLASSES
# FEW_SHOT_CLASSES_3
# ZERO_SHOT_CLASSES_3


load_from = ('work_dirs/asd_65_8_7_r101_fpn_coco_10shot-fine-tuning_for_fs_set3/base_model_random_init_bbox_head_for_fs_set3.pth')

evaluation = dict(
    interval=4000,
    metric='bbox',
    classwise=True,
    class_splits=['BASE_CLASSES', 'FEW_SHOT_CLASSES_3', 'ZERO_SHOT_CLASSES_3']
    )

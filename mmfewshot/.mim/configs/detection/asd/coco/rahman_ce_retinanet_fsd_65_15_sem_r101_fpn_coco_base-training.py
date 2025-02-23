_base_ = [
    '../../_base_/datasets/any_shot_detection/base_coco_65_15.py',
    './retinanet_fsd_65_15_sem_r101_fpn.py',
]


# model settings
model = dict(
    bbox_head=dict(num_classes=65))
evaluation = dict(interval=100000, metric='bbox', classwise=True)

##  no inherit from base config of default runtime
## ---------------------------------------
checkpoint_config = dict(interval=100000)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
use_infinite_sampler = True
# a magical seed works well in most cases for this repo!!!
# using different seeds might raise some issues about reproducibility
seed = 42
## ---------------------------------------

## no inherit from base config of schedule
## ---------------------------------------
# optimizer
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(type='Adam', lr=0.00001,  weight_decay=0.0001) # for single gpu
optimizer = dict(type='Adam', lr=0.00001,  weight_decay=0.0001) 
optimizer_config = dict(grad_clip=dict(max_norm=0.001, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[500000],
    )
runner = dict(type='IterBasedRunner', max_iters=500000) # for focal loss

## ---------------------------------------
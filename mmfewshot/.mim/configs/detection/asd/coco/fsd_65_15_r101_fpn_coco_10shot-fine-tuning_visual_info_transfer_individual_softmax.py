_base_ = [
    '../../_base_/datasets/any_shot_detection/few_shot_coco_65_15.py',
    '../../_base_/schedules/schedule.py', './fsd_65_15_r101_fpn.py',
    '../../_base_/default_runtime.py'
]


evaluation = dict(interval=80000)
checkpoint_config = dict(interval=80000)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=10, step=[144000])
runner = dict(max_iters=160000)
model = dict(
    roi_head=dict(
        type='StandardRoIHead_visual_info_transfer_individual_softmax',
        bbox_head=dict(
            type='Shared2FCBBoxHead_visual_info_transfer_individual_softmax',
            )
    )
)

# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.
# load_from = ('work_dirs/asd_r101_fpn_coco_base-training/'
#              'base_model_random_init_bbox_head.pth')

load_from = ('work_dirs/fsd_65_15_r101_fpn_coco_base-training/'
             'base_model_random_init_bbox_head.pth')
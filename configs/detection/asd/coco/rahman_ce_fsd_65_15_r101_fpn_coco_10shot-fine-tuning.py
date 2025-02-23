_base_ = [
    '../../_base_/datasets/any_shot_detection/few_shot_coco_65_15.py',
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
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(
        bbox_head=dict(
            num_classes=80,
    ))
)

# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.
load_from = ('work_dirs/asd_r101_fpn_coco_base-training/'
             'base_model_random_init_bbox_head.pth')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        save_dataset=True,
        type='AnyShotCocoDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file='data/any_shot_ann/coco/annotations/rahman_fsd_finetune_dataset/rahman_fsd_full_box_10shot_trainval.json'
                )
        ],
        num_novel_shots=None,
        num_base_shots=None,
        classes='ALL_CLASSES',
        instance_wise=False),
    )
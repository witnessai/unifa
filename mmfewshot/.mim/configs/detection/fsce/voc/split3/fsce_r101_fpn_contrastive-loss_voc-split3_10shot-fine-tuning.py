_base_ = [
    '../../../_base_/datasets/fine_tune_based/few_shot_voc.py',
    '../../../_base_/schedules/schedule.py',
    '../../fsce_r101_fpn_contrastive_loss.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        type='FewShotVOCDefaultDataset',
        ann_cfg=[dict(method='FSCE', setting='SPLIT3_10SHOT')],
        num_novel_shots=10,
        num_base_shots=10,
        classes='ALL_CLASSES_SPLIT3'),
    val=dict(classes='ALL_CLASSES_SPLIT3'),
    test=dict(classes='ALL_CLASSES_SPLIT3'))
evaluation = dict(
    interval=7500,
    class_splits=['BASE_CLASSES_SPLIT3', 'NOVEL_CLASSES_SPLIT3'])
checkpoint_config = dict(interval=7500)
optimizer = dict(lr=0.001)
lr_config = dict(warmup_iters=200, gamma=0.5, step=[8000, 13000])
runner = dict(max_iters=15000)
custom_hooks = [
    dict(
        type='ContrastiveLossDecayHook',
        decay_steps=(6000, 10000),
        decay_rate=0.5)
]
model = dict(
    roi_head=dict(
        bbox_head=dict(
            with_weight_decay=True,
            loss_contrast=dict(iou_threshold=0.8, loss_weight=0.5))))
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/fsce/README.md for more details.
load_from = ('work_dirs/fsce_r101_fpn_voc-split3_base-training/'
             'base_model_random_init_bbox_head.pth')

_base_ = [
    '../../../_base_/datasets/fine_tune_based/base_voc.py',
    '../../../_base_/schedules/schedule.py',
    '../../../_base_/models/faster_rcnn_r50_caffe_fpn.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
data = dict(
    train=dict(classes='BASE_CLASSES_SPLIT1'),
    val=dict(classes='BASE_CLASSES_SPLIT1'),
    test=dict(classes='BASE_CLASSES_SPLIT1'))
lr_config = dict(warmup_iters=100, step=[12000, 16000])
runner = dict(max_iters=18000)
# model settings
model = dict(
    type='TFA',
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    roi_head=dict(bbox_head=dict(num_classes=15)))
# using regular sampler can get a better base model
use_infinite_sampler = False

# morjio modify
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

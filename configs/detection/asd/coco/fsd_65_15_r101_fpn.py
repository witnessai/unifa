# refer to tfa_r101_fpn.py
_base_ = ['/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/configs/detection/_base_/models/faster_rcnn_r50_caffe_fpn.py']
model = dict(
    frozen_parameters=[
        'backbone', 'neck', 'rpn_head', 'roi_head.bbox_head.shared_fcs'
    ],
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101, frozen_stages=4),
            
)

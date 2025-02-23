# Copyright (c) OpenMMLab. All rights reserved.
import torch 
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler, bbox_overlaps
from mmdet.models import builder
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.base import BaseDetector
from mmdet.models.detectors.test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin
from mmdet.models.detectors.two_stage import TwoStageDetector
import ipdb


@DETECTORS.register_module()
class FasterSemanticRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FasterSemanticRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
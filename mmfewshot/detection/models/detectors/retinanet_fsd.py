# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from .single_stage_fsd import SingleStageDetector_fsd


@DETECTORS.register_module()
class RetinaNet_fsd(SingleStageDetector_fsd):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RetinaNet_fsd, self).__init__(backbone, neck, bbox_head, train_cfg,
                                        test_cfg, pretrained, init_cfg)
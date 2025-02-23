# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from .anchor_head_fsd import AnchorHead_fsd
from mmdet.core import (anchor_inside_flags, unmap)

from mmfewshot.detection.models.losses import LSoftmaxLinear

import numpy as np
import ipdb 
import time 


@HEADS.register_module()
class RetinaSemHead(AnchorHead_fsd):
    r"""An anchor-based head used in `RetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Example:
        >>> import torch
        >>> self = RetinaHead(11, 7)
        >>> x = torch.rand(1, 7, 32, 32)
        >>> cls_score, bbox_pred = self.forward_single(x)
        >>> # Each anchor predicts a score for each class except background
        >>> cls_per_anchor = cls_score.shape[1] / self.num_anchors
        >>> box_per_anchor = bbox_pred.shape[1] / self.num_anchors
        >>> assert cls_per_anchor == (self.num_classes)
        >>> assert box_per_anchor == 4
    """

    def __init__(self,
                num_classes,
                in_channels,
                feat_channels=256,
                stacked_convs=4,
                conv_cfg=None,
                norm_cfg=None,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    octave_base_scale=4,
                    scales_per_octave=3,
                    ratios=[0.5, 1.0, 2.0],
                    strides=[8, 16, 32, 64, 128]),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=True,
                    target_means=(.0, .0, .0, .0),
                    target_stds=(1.0, 1.0, 1.0, 1.0)),
                reg_decoded_bbox=False,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0),
                loss_bbox=dict(
                    type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                train_cfg=None,
                test_cfg=None,
                init_cfg=dict(
                    type='Normal',
                    layer='Conv2d', 
                    std=0.01,
                    override=[
                        dict(
                        type='Constant',
                        name='conv_semantic',
                        val=0,
                        bias_prob=0.01,
                        ),
                        dict(
                        type='Uniform',
                        name='kernel_semantic',
                        a = -0.05,
                        b = 0.05,
                        # bias_prob=0.50125
                        ),
                    ]
                ),
                with_semantic=True,
                semantic_dims=300,
                reg_with_semantic=False,
                share_semantic=False,
                voc_path=None,
                vec_path=None,
                use_lsoftmax=False,
                **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.with_semantic = with_semantic 
        self.semantic_dims = semantic_dims
        self.reg_with_semantic = reg_with_semantic
        self.share_semantic = share_semantic
        self.voc_path = voc_path
        self.vec_path = vec_path
        self.use_lsoftmax = use_lsoftmax 

        if self.with_semantic:
            if voc_path is not None:
                voc = np.loadtxt(voc_path, dtype='float32', delimiter=',')
            else:
                voc = None
            vec_load = np.loadtxt(vec_path, dtype='float32', delimiter=',')
            
            ## use bg embedding
            # vec = np.concatenate([vec_load[:, 1:num_classes+1], vec_load[:, 0:1]], axis=1)
            # vec_unseen = np.concatenate([vec_load[:, num_classes+1:], vec_load[:, 0:1]], axis=1)

            ## not use bg embedding
            vec = vec_load[:, 1:num_classes+1]
            ## note:


            vec_unseen = vec_load[:, num_classes+1:]

            vec = torch.tensor(vec, dtype=torch.float32)
            if voc is not None:
                voc = torch.tensor(voc, dtype=torch.float32)
                self.voc = voc.cuda() # 300*4717
            vec_unseen = torch.tensor(vec_unseen, dtype=torch.float32)
            self.vec = vec.cuda()
            self.vec_unseen = vec_unseen.cuda()
                
            if self.use_lsoftmax:
                self.lsoftmax = LSoftmaxLinear(num_classes, num_classes, margin=4)
        
            
        super(RetinaSemHead, self).__init__(
            num_classes,
            in_channels,
            feat_channels=feat_channels,
            anchor_generator=anchor_generator,
            bbox_coder=bbox_coder,
            reg_decoded_bbox=reg_decoded_bbox,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **kwargs)

    # call by super class in __init__
    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_base_priors * 4, 3, padding=1)
    
        if self.with_semantic:
            self.d = 300
            self.conv_semantic = nn.Conv2d(
                                    self.feat_channels,
                                    self.num_base_priors * self.d,
                                    3,
                                    padding=1)
            if self.voc is not None:
                self.kernel_semantic = nn.Linear(self.voc.shape[1], self.vec.shape[0]) #n*300
            else:
                
                self.kernel_semantic = nn.Linear(self.vec.shape[1], self.vec.shape[1])
        else:
            self.retina_cls = nn.Conv2d(
                    self.feat_channels,
                    self.num_base_priors * self.cls_out_channels,
                    3,
                    padding=1)

    def forward_single(self, x):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_anchors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_anchors * 4.
        """
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        
        if self.with_semantic:
            # ipdb.set_trace()
            # semantic_feature.shape is [2, 2700, h, w]
            semantic_feature = self.conv_semantic(cls_feat)
            img_num, h, w = semantic_feature.shape[0], semantic_feature.shape[2], semantic_feature.shape[3]
            # semantic_feature.shape is [img_num * h * w * 9, 300], 9 is the anchor_num per location
            semantic_feature = semantic_feature.view(-1, 300)
            
            if self.voc is not None:
                # self.vec.shape is [300, 65], self.vo.shape is [300, 4717]
                semantic_score = torch.mm(self.kernel_semantic(self.voc), self.vec)
            else:
                semantic_score = self.kernel_semantic(self.vec)
            
            semantic_score = torch.tanh(semantic_score) # semantic_score.shape is [300, 65]
            # the left semantic_score.shape is [img_num * h * w * 9, 65]
            semantic_score = torch.mm(semantic_feature, semantic_score) 
            # cls_score.shape is [img_num, num_class * 9, h, w]
            cls_score = semantic_score.view(img_num, -1, h, w)
        else:
            cls_score = self.retina_cls(cls_feat)
        # print(self.kernel_semantic(self.vec))
        # print(semantic_feature)
        # print(cls_score.shape)
        
        bbox_pred = self.retina_reg(reg_feat)
    
        
        return cls_score, bbox_pred

    def _get_targets_single(self,
                            flat_anchors,
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.
        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors ,4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            img_meta (dict): Meta info of the image.
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.
        Returns:
            tuple:
                labels_list (list[Tensor]): Labels of each level
                label_weights_list (list[Tensor]): Label weights of each level
                bbox_targets_list (list[Tensor]): BBox targets of each level
                bbox_weights_list (list[Tensor]): BBox weights of each level
                num_total_pos (int): Number of positive samples in all images
                num_total_neg (int): Number of negative samples in all images
        """
        # the last param self.train_cfg.allowed_border is set to -1
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        assign_result = self.assigner.assign(
            anchors, gt_bboxes, gt_bboxes_ignore,
            None if self.sampling else gt_labels)
        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)
        
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)
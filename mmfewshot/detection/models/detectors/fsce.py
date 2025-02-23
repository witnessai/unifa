# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.two_stage import TwoStageDetector

# modified
import numpy as np
import torch
import ipdb 
from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
import warnings
# from mmdet.core.visualization import imshow_det_bboxes
import mmcv
# follow mmdetection/mmdet/core/visualization/image.py  setting 
import cv2
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
# from mmdet.core.evaluation.panoptic_utils import INSTANCE_OFFSET
# from mmdet.core.visualization import get_palette, palette_val
# from mmdet.core.visualization import draw_labels, draw_bboxes, draw_masks, _get_adaptive_scales
from mmdet.core.utils import mask2ndarray
# from mmdet.core.mask.structures import bitmap_to_polygon
from mmdet.core.bbox.iou_calculators import build_iou_calculator

@DETECTORS.register_module()
class FSCE(TwoStageDetector):
    """Implementation of `FSCE <https://arxiv.org/abs/2103.05950>`_"""
    def feats_extract(self,
                      img,
                      img_meta,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        
        img = img.cuda()
        x = self.extract_feat(img)


        # RPN forward and loss
        if self.with_rpn:

            # rpn_outs = self.rpn_head(x)
            # rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                        #   self.train_cfg.rpn)
            # rpn_losses = self.rpn_head.loss(
            #     *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            # losses.update(rpn_losses)
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            gt_bboxes = [x.cuda() for x in gt_bboxes]
            proposal_list = self.rpn_head.simple_test_rpn(x, img_meta)
            # rpn_losses, proposal_list = self.rpn_head.forward_train(
            #     x,
            #     img_meta,
            #     gt_bboxes,
            #     gt_labels=None,
            #     gt_bboxes_ignore=gt_bboxes_ignore,
            #     proposal_cfg=proposal_cfg)
            
            # proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            
            # proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
        else:
            proposal_list = proposals

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            sampling_iou_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i].cuda(),
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i].cuda())
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i].cuda(),
                    gt_labels[i].cuda(),
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                
                sampling_iou_result = torch.cat([assign_result.max_overlaps[sampling_result.pos_inds], assign_result.max_overlaps[sampling_result.neg_inds]])
                sampling_results.append(sampling_result)
                sampling_iou_results.append(sampling_iou_result)

        # bbox head forward and loss

        # if self.with_bbox:
        rois = bbox2roi([res.bboxes for res in sampling_results]) # rois.shape is [2048, 5]
        # TODO: a more flexible way to decide which feature maps to use
        # bbox_feats.shape is [2048, 1024]
        bbox_feats = self.roi_head.bbox_roi_extractor(
            x[:self.roi_head.bbox_roi_extractor.num_inputs], rois) 
        
        # bbox_feats-->shape ==> num_boxes x 5
        if self.with_shared_head: # False
            bbox_feats = self.shared_head(bbox_feats)
        # cls_score, bbox_pred = self.bbox_head(bbox_feats)
        # aaa = self.classfier(bbox_feats)

        num_shared_fcs = 2
        # if self.bbox_head.num_shared_fcs > 0: # self.bbox_head.num_shared_fcs is 2
        if num_shared_fcs > 0:
            # already avg_pooled 
            # if self.with_avg_pool:
            #     x = self.avg_pool(x)
            bbox_feats = bbox_feats.view(bbox_feats.size(0), -1)
            for fc in self.roi_head.bbox_head.shared_fcs:
                bbox_feats = self.roi_head.bbox_head.relu(fc(bbox_feats))

        # len(gt_bboxes) is 4, gt_bboxes[i].shape is [gt_num_of_img_i, 4]
        # len(gt_labels) is 4, gt_labels[i].shape is [gt_num_of_img_i]
        # ipdb.set_trace()
        sampling_iou_results = torch.cat(sampling_iou_results)
        bbox_targets = self.roi_head.bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
        K = self.roi_head.bbox_head.num_classes
        bg_inds = np.where(bbox_targets[0].data.cpu().numpy()==K)[0]
        fg_inds = np.where(bbox_targets[0].data.cpu().numpy()<K)[0]
        # print(sampling_result.pos_inds)
        # print(fg_inds)
        # print(sampling_result.neg_inds)
        # print(bg_inds)
        # ipdb.set_trace()
        #bg_scores = cls_score[:, 0]
        #sorted_args = np.argsort(bg_scores.data.cpu().numpy(), kind='mergesort')[:len(fg_inds)*3]
        #selected_bg_inds = np.intersect1d(sorted_args, bg_inds)
        sub_neg_inds = np.random.permutation(bg_inds)[:int(2*len(fg_inds))]
        # 
        inds_to_select = np.concatenate((sub_neg_inds, fg_inds))
        return bbox_feats[inds_to_select], bbox_targets[0][inds_to_select], bbox_targets[2][inds_to_select]
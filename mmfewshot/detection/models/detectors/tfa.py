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
class TFA(TwoStageDetector):
    """Implementation of `TFA <https://arxiv.org/abs/2003.06957>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(TFA, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if 'asd' in test_cfg.rcnn:
            self.roi_head.bbox_head.gasd = test_cfg.rcnn.gasd
            self.roi_head.bbox_head.asd = test_cfg.rcnn.asd

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
        # return bbox_feats[inds_to_select], bbox_targets[0][inds_to_select], bbox_targets[2][inds_to_select], sampling_iou_results[inds_to_select]
        # return bbox_feats, bbox_targets[0], bbox_targets[2]


    def multi_label_feats_extract(self,
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
        iou_calculator = dict(type='BboxOverlaps2D')
        self.iou_calculator = build_iou_calculator(iou_calculator)
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            sampling_iou_results = []
            sampling_soft_label_results = []
            sampling_gt_label_results = []
            # ipdb.set_trace()
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
                # assign_result.max_overlaps.shape is [1001]
                gt_bboxes_with_score = torch.cat([gt_bboxes[i].cuda(), torch.ones(gt_bboxes[i].shape[0], 1).cuda()], 1)
                proposal_list_with_gt = torch.cat([gt_bboxes_with_score, proposal_list[i]])
                overlaps = self.iou_calculator(proposal_list_with_gt, gt_bboxes[i].cuda())
                sampling_soft_label_result = torch.cat([overlaps[sampling_result.pos_inds], overlaps[sampling_result.neg_inds]])
                sampling_iou_result = torch.cat([assign_result.max_overlaps[sampling_result.pos_inds], assign_result.max_overlaps[sampling_result.neg_inds]])
                sampling_gt_label_result = gt_labels[i].cuda().repeat(len(sampling_iou_result), 1)
                sampling_results.append(sampling_result)
                sampling_iou_results.append(sampling_iou_result)
                sampling_soft_label_results.append(sampling_soft_label_result)
                sampling_gt_label_results.append(sampling_gt_label_result)
                ipdb.set_trace()
                # if gt_bboxes[i].shape[0] > 1: ipdb.set_trace()
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
        
        
        bbox_targets = self.roi_head.bbox_head.get_targets(sampling_results, gt_bboxes, gt_labels, self.train_cfg.rcnn)
        sampling_iou_results = torch.cat(sampling_iou_results)
        # img_one_gt_num = sampling_soft_label_results[0].shape[1]
        # img_two_gt_num = sampling_soft_label_results[1].shape[1]
        no_sampling_gt_labels = []
        max_gt_num = -1
        for gt_label in gt_labels:
            no_sampling_gt_labels.append(gt_label.unsqueeze(0))
            if gt_label.shape[0] > max_gt_num:
                max_gt_num = gt_label.shape[0]
        for i, gt_label in enumerate(gt_labels):
            diff = max_gt_num - gt_label.shape[0]
            if diff != 0:
                length = sampling_soft_label_results[i].shape[0]
                supp = torch.ones(length, diff) * -1 
                sampling_soft_label_results[i] = torch.cat([sampling_soft_label_results[i], supp.cuda()], 1)
                supp = torch.ones(1, diff) * -1
                no_sampling_gt_labels[i] = torch.cat([no_sampling_gt_labels[i], supp], 1)

        # if img_one_gt_num != img_two_gt_num:
        #     if img_one_gt_num > img_two_gt_num:
        #         diff = img_one_gt_num - img_two_gt_num
        #         supp = torch.ones(512, diff) * -1
        #         sampling_soft_label_results[1] = torch.cat([sampling_soft_label_results[1], supp.cuda()], 1)
        #         supp = torch.ones(1, diff) * -1
        #         no_sampling_gt_labels[1] = torch.cat([no_sampling_gt_labels[1], supp], 1)
        #     elif img_one_gt_num < img_two_gt_num:
        #         diff = img_two_gt_num - img_one_gt_num
        #         supp = torch.ones(512, diff) * -1
        #         sampling_soft_label_results[0] = torch.cat([sampling_soft_label_results[0], supp.cuda()], 1)
        #         supp = torch.ones(1, diff) * -1
        #         no_sampling_gt_labels[0] = torch.cat([no_sampling_gt_labels[0], supp], 1)
        
        sampling_soft_label_results = torch.cat(sampling_soft_label_results)
        sampling_gt_label_results = torch.cat(sampling_gt_label_results)
        no_sampling_gt_labels = torch.cat(no_sampling_gt_labels)
        no_sampling_gt_labels = no_sampling_gt_labels.cuda()
        # print(no_sampling_gt_labels)
        # ipdb.set_trace()
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
        return bbox_feats[inds_to_select], bbox_targets[0][inds_to_select], bbox_targets[2][inds_to_select], sampling_iou_results[inds_to_select], sampling_soft_label_results[inds_to_select], sampling_gt_label_results[inds_to_select], no_sampling_gt_labels
        # return bbox_feats, bbox_targets[0], bbox_targets[2]
        # bbox_targets[0] represents label
        # bbox_targets[2] represents bbox coor.


    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.
        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            bbox_result, segm_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        img = asd_imshow_det_bboxes(
            img,
            bboxes,
            labels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img


def imshow_det_bboxes(img,
                    bboxes=None,
                    labels=None,
                    segms=None,
                    class_names=None,
                    score_thr=0,
                    bbox_color='green',
                    text_color='green',
                    mask_color=None,
                    thickness=2,
                    font_size=8,
                    win_name='',
                    show=True,
                    wait_time=0,
                    out_file=None):
    """Draw bboxes and class labels (with scores) on an image.
    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray | None): Masks, shaped (n,h,w) or None.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.
        bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
        If a single color is given, it will be applied to all classes.
        The tuple of color should be in RGB order. Default: 'green'.
        text_color (list[tuple] | tuple | str | None): Colors of texts.
        If a single color is given, it will be applied to all classes.
        The tuple of color should be in RGB order. Default: 'green'.
        mask_color (list[tuple] | tuple | str | None, optional): Colors of
        masks. If a single color is given, it will be applied to all
        classes. The tuple of color should be in RGB order.
        Default: None.
        thickness (int): Thickness of lines. Default: 2.
        font_size (int): Font size of texts. Default: 13.
        show (bool): Whether to show the image. Default: True.
        win_name (str): The window name. Default: ''.
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None.
    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes is None or bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
        'labels.shape[0] should not be less than bboxes.shape[0].'
    assert segms is None or segms.shape[0] == labels.shape[0], \
        'segms.shape[0] and labels.shape[0] should have the same length.'
    assert segms is not None or bboxes is not None, \
        'segms and bboxes should not be None at the same time.'

    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    max_label = int(max(labels) if len(labels) > 0 else 0)
    text_palette = palette_val(get_palette(text_color, max_label + 1))
    text_colors = [text_palette[label] for label in labels]

    num_bboxes = 0
    if bboxes is not None:
        num_bboxes = bboxes.shape[0]
        bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
        colors = [bbox_palette[label] for label in labels[:num_bboxes]]
        draw_bboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)

        horizontal_alignment = 'left'
        positions = bboxes[:, :2].astype(np.int32) + thickness
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        scales = _get_adaptive_scales(areas)
        scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None
        draw_labels(
            ax,
            labels[:num_bboxes],
            positions,
            scores=scores,
            class_names=class_names,
            color=text_colors,
            font_size=font_size,
            scales=scales,
            horizontal_alignment=horizontal_alignment)

    if segms is not None:
        mask_palette = get_palette(mask_color, max_label + 1)
        colors = [mask_palette[label] for label in labels]
        colors = np.array(colors, dtype=np.uint8)
        draw_masks(ax, img, segms, colors, with_edge=True)

        if num_bboxes < segms.shape[0]:
            segms = segms[num_bboxes:]
            horizontal_alignment = 'center'
            areas = []
            positions = []
            for mask in segms:
                _, _, stats, centroids = cv2.connectedComponentsWithStats(
                    mask.astype(np.uint8), connectivity=8)
                largest_id = np.argmax(stats[1:, -1]) + 1
                positions.append(centroids[largest_id])
                areas.append(stats[largest_id, -1])
            areas = np.stack(areas, axis=0)
            scales = _get_adaptive_scales(areas)
            draw_labels(
                ax,
                labels[num_bboxes:],
                positions,
                class_names=class_names,
                color=text_colors,
                font_size=font_size,
                scales=scales,
                horizontal_alignment=horizontal_alignment)

    plt.imshow(img)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img


def asd_imshow_det_bboxes(img,
                    bboxes=None,
                    labels=None,
                    segms=None,
                    class_names=None,
                    score_thr=0,
                    bbox_color='green',
                    text_color='green',
                    mask_color=None,
                    thickness=2,
                    font_size=8,
                    win_name='',
                    show=True,
                    wait_time=0,
                    out_file=None):
    """Draw bboxes and class labels (with scores) on an image.
    Args:
        img (str | ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        segms (ndarray | None): Masks, shaped (n,h,w) or None.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown. Default: 0.
        bbox_color (list[tuple] | tuple | str | None): Colors of bbox lines.
        If a single color is given, it will be applied to all classes.
        The tuple of color should be in RGB order. Default: 'green'.
        text_color (list[tuple] | tuple | str | None): Colors of texts.
        If a single color is given, it will be applied to all classes.
        The tuple of color should be in RGB order. Default: 'green'.
        mask_color (list[tuple] | tuple | str | None, optional): Colors of
        masks. If a single color is given, it will be applied to all
        classes. The tuple of color should be in RGB order.
        Default: None.
        thickness (int): Thickness of lines. Default: 2.
        font_size (int): Font size of texts. Default: 13.
        show (bool): Whether to show the image. Default: True.
        win_name (str): The window name. Default: ''.
        wait_time (float): Value of waitKey param. Default: 0.
        out_file (str, optional): The filename to write the image.
            Default: None.
    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    assert bboxes is None or bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
        'labels.shape[0] should not be less than bboxes.shape[0].'
    assert segms is None or segms.shape[0] == labels.shape[0], \
        'segms.shape[0] and labels.shape[0] should have the same length.'
    assert segms is not None or bboxes is not None, \
        'segms and bboxes should not be None at the same time.'

    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        zero_shot_score_thr = score_thr - 0.1
        non_zero_shot_score_thr = score_thr + 0.1
        inds = scores > score_thr
        zero_shot_id_set = [4, 15, 28, 29, 48, 61, 64]
        non_zero_shot_id_set = list(set(list(range(80)))-set(zero_shot_id_set))
        zero_shot_inds = np.array([x in zero_shot_id_set for x in labels])
        non_zero_shot_inds = np.array([x in non_zero_shot_id_set for x in labels])
        non_zero_shot_valid = scores[non_zero_shot_inds] > non_zero_shot_score_thr
        zero_shot_valid = scores[zero_shot_inds] > zero_shot_score_thr
        bboxes = np.concatenate((bboxes[non_zero_shot_inds][non_zero_shot_valid], bboxes[zero_shot_inds][zero_shot_valid]))
        labels = np.concatenate((labels[non_zero_shot_inds][non_zero_shot_valid], labels[zero_shot_inds][zero_shot_valid]))
        # ipdb.set_trace()
        # (scores[non_zero_shot_inds] > non_zero_shot_score_thr)
        # bboxes = bboxes[inds, :]
        # labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    EPS = 1e-2
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    max_label = int(max(labels) if len(labels) > 0 else 0)
    text_palette = palette_val(get_palette(text_color, max_label + 1))
    text_colors = [text_palette[label] for label in labels]

    num_bboxes = 0
    if bboxes is not None:
        num_bboxes = bboxes.shape[0]
        bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
        
        # colors = [bbox_palette[label] for label in labels[:num_bboxes]]
        colors_flag = [label in zero_shot_id_set for label in labels[:num_bboxes]]
        colors = []
        for f in colors_flag:
            if f is True:
                # colors.append((0.2823529411764706, 0.396078431372549, 0.9450980392156862))
                colors.append((1, 0, 0))
            else:
                colors.append((0, 1, 0))
                # colors.append((0.2823529411764706, 0.2823529411764706, 0.2823529411764706))
        
        draw_bboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)

        horizontal_alignment = 'left'
        positions = bboxes[:, :2].astype(np.int32) + thickness
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        scales = _get_adaptive_scales(areas)
        scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None
        draw_labels(
            ax,
            labels[:num_bboxes],
            positions,
            scores=scores,
            class_names=class_names,
            color=colors,
            # color=text_colors,
            font_size=font_size,
            scales=scales,
            horizontal_alignment=horizontal_alignment)

    if segms is not None:
        mask_palette = get_palette(mask_color, max_label + 1)
        colors = [mask_palette[label] for label in labels]
        colors = np.array(colors, dtype=np.uint8)
        draw_masks(ax, img, segms, colors, with_edge=True)

        if num_bboxes < segms.shape[0]:
            segms = segms[num_bboxes:]
            horizontal_alignment = 'center'
            areas = []
            positions = []
            for mask in segms:
                _, _, stats, centroids = cv2.connectedComponentsWithStats(
                    mask.astype(np.uint8), connectivity=8)
                largest_id = np.argmax(stats[1:, -1]) + 1
                positions.append(centroids[largest_id])
                areas.append(stats[largest_id, -1])
            areas = np.stack(areas, axis=0)
            scales = _get_adaptive_scales(areas)
            draw_labels(
                ax,
                labels[num_bboxes:],
                positions,
                class_names=class_names,
                color=text_colors,
                font_size=font_size,
                scales=scales,
                horizontal_alignment=horizontal_alignment)

    plt.imshow(img)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img




def palette_val(palette):
    """Convert palette to matplotlib palette.
    Args:
        palette List[tuple]: A list of color tuples.
    Returns:
        List[tuple[float]]: A list of RGB matplotlib color tuples.
    """
    new_palette = []
    for color in palette:
        color = [c / 255 for c in color]
        new_palette.append(tuple(color))
    return new_palette


def get_palette(palette, num_classes):
    """Get palette from various inputs.
    Args:
        palette (list[tuple] | str | tuple | :obj:`Color`): palette inputs.
        num_classes (int): the number of classes.
    Returns:
        list[tuple[int]]: A list of color tuples.
    """
    assert isinstance(num_classes, int)

    if isinstance(palette, list):
        dataset_palette = palette
    elif isinstance(palette, tuple):
        dataset_palette = [palette] * num_classes
    elif palette == 'random' or palette is None:
        state = np.random.get_state()
        # random color
        np.random.seed(42)
        palette = np.random.randint(0, 256, size=(num_classes, 3))
        np.random.set_state(state)
        dataset_palette = [tuple(c) for c in palette]
    elif palette == 'coco':
        from mmdet.datasets import CocoDataset, CocoPanopticDataset
        dataset_palette = CocoDataset.PALETTE
        if len(dataset_palette) < num_classes:
            dataset_palette = CocoPanopticDataset.PALETTE
    elif palette == 'citys':
        from mmdet.datasets import CityscapesDataset
        dataset_palette = CityscapesDataset.PALETTE
    elif palette == 'voc':
        from mmdet.datasets import VOCDataset
        dataset_palette = VOCDataset.PALETTE
    elif mmcv.is_str(palette):
        dataset_palette = [mmcv.color_val(palette)[::-1]] * num_classes
    else:
        raise TypeError(f'Invalid type for palette: {type(palette)}')

    assert len(dataset_palette) >= num_classes, \
        'The length of palette should not be less than `num_classes`.'
    return dataset_palette

def _get_adaptive_scales(areas, min_area=800, max_area=30000):
    """Get adaptive scales according to areas.
    The scale range is [0.5, 1.0]. When the area is less than
    ``'min_area'``, the scale is 0.5 while the area is larger than
    ``'max_area'``, the scale is 1.0.
    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Default: 800.
        max_area (int): Upper bound areas for adaptive scales.
            Default: 30000.
    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


def _get_bias_color(base, max_dist=30):
    """Get different colors for each masks.
    Get different colors for each masks by adding a bias
    color to the base category color.
    Args:
        base (ndarray): The base category color with the shape
            of (3, ).
        max_dist (int): The max distance of bias. Default: 30.
    Returns:
        ndarray: The new color for a mask with the shape of (3, ).
    """
    new_color = base + np.random.randint(
        low=-max_dist, high=max_dist + 1, size=3)
    return np.clip(new_color, 0, 255, new_color)


def draw_bboxes(ax, bboxes, color='g', alpha=0.8, thickness=2):
    """Draw bounding boxes on the axes.
    Args:
        ax (matplotlib.Axes): The input axes.
        bboxes (ndarray): The input bounding boxes with the shape
            of (n, 4).
        color (list[tuple] | matplotlib.color): the colors for each
            bounding boxes.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
        thickness (int): Thickness of lines. Default: 2.
    Returns:
        matplotlib.Axes: The result axes.
    """
    polygons = []
    for i, bbox in enumerate(bboxes):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
    p = PatchCollection(
        polygons,
        facecolor='none',
        edgecolors=color,
        linewidths=thickness,
        alpha=alpha)
    ax.add_collection(p)

    return ax


def draw_labels(ax,
                labels,
                positions,
                scores=None,
                class_names=None,
                color='w',
                font_size=8,
                scales=None,
                horizontal_alignment='left'):
    """Draw labels on the axes.
    Args:
        ax (matplotlib.Axes): The input axes.
        labels (ndarray): The labels with the shape of (n, ).
        positions (ndarray): The positions to draw each labels.
        scores (ndarray): The scores for each labels.
        class_names (list[str]): The class names.
        color (list[tuple] | matplotlib.color): The colors for labels.
        font_size (int): Font size of texts. Default: 8.
        scales (list[float]): Scales of texts. Default: None.
        horizontal_alignment (str): The horizontal alignment method of
            texts. Default: 'left'.
    Returns:
        matplotlib.Axes: The result axes.
    """
    for i, (pos, label) in enumerate(zip(positions, labels)):
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        if scores is not None:
            label_text += f'|{scores[i]:.02f}'
        text_color = color[i] if isinstance(color, list) else color

        font_size_mask = font_size if scales is None else font_size * scales[i]
        ax.text(
            pos[0],
            pos[1],
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            fontsize=font_size_mask,
            verticalalignment='top',
            horizontalalignment=horizontal_alignment)

    return ax


def draw_masks(ax, img, masks, color=None, with_edge=True, alpha=0.8):
    """Draw masks on the image and their edges on the axes.
    Args:
        ax (matplotlib.Axes): The input axes.
        img (ndarray): The image with the shape of (3, h, w).
        masks (ndarray): The masks with the shape of (n, h, w).
        color (ndarray): The colors for each masks with the shape
            of (n, 3).
        with_edge (bool): Whether to draw edges. Default: True.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
    Returns:
        matplotlib.Axes: The result axes.
        ndarray: The result image.
    """
    taken_colors = set([0, 0, 0])
    if color is None:
        random_colors = np.random.randint(0, 255, (masks.size(0), 3))
        color = [tuple(c) for c in random_colors]
        color = np.array(color, dtype=np.uint8)
    polygons = []
    for i, mask in enumerate(masks):
        if with_edge:
            contours, _ = bitmap_to_polygon(mask)
            polygons += [Polygon(c) for c in contours]

        color_mask = color[i]
        while tuple(color_mask) in taken_colors:
            color_mask = _get_bias_color(color_mask)
        taken_colors.add(tuple(color_mask))

        mask = mask.astype(bool)
        img[mask] = img[mask] * (1 - alpha) + color_mask * alpha

    p = PatchCollection(
        polygons, facecolor='none', edgecolors='w', linewidths=1, alpha=0.8)
    ax.add_collection(p)

    return ax, img

def bitmap_to_polygon(bitmap):
    """Convert masks from the form of bitmaps to polygons.
    Args:
        bitmap (ndarray): masks in bitmap representation.
    Return:
        list[ndarray]: the converted mask in polygon representation.
        bool: whether the mask has holes.
    """
    bitmap = np.ascontiguousarray(bitmap).astype(np.uint8)
    # cv2.RETR_CCOMP: retrieves all of the contours and organizes them
    #   into a two-level hierarchy. At the top level, there are external
    #   boundaries of the components. At the second level, there are
    #   boundaries of the holes. If there is another contour inside a hole
    #   of a connected component, it is still put at the top level.
    # cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
    outs = cv2.findContours(bitmap, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours = outs[-2]
    hierarchy = outs[-1]
    if hierarchy is None:
        return [], False
    # hierarchy[i]: 4 elements, for the indexes of next, previous,
    # parent, or nested contours. If there is no corresponding contour,
    # it will be -1.
    with_hole = (hierarchy.reshape(-1, 4)[:, 3] >= 0).any()
    contours = [c.reshape(-1, 2) for c in contours]
    return contours, with_hole
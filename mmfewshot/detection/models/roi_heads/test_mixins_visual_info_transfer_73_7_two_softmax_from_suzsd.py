# Copyright (c) OpenMMLab. All rights reserved.
import sys
import warnings

import numpy as np
import torch

from mmdet.core import (bbox2roi, bbox_mapping, merge_aug_bboxes,
                        merge_aug_masks, multiclass_nms)

import ipdb 
import torch.nn.functional as F

if sys.version_info >= (3, 7):
    from mmdet.utils.contextmanagers import completed

from mmfewshot.detection.models.roi_heads.splits import get_asd_zero_shot_class_ids, get_seen_class_ids
# from splits import get_asd_zero_shot_class_ids, get_seen_class_ids

import time 

cls_rel_count_matrix = torch.zeros(81, 81)

class BBoxTestMixin_visual_info_transfer_73_7_two_softmax_from_suzsd: 

    if sys.version_info >= (3, 7):

        async def async_test_bboxes(self,
                                    x,
                                    img_metas,
                                    proposals,
                                    rcnn_test_cfg,
                                    rescale=False,
                                    **kwargs):
            """Asynchronized test for box head without augmentation."""
            rois = bbox2roi(proposals)
            roi_feats = self.bbox_roi_extractor(
                x[:len(self.bbox_roi_extractor.featmap_strides)], rois)
            if self.with_shared_head:
                roi_feats = self.shared_head(roi_feats)
            sleep_interval = rcnn_test_cfg.get('async_sleep_interval', 0.017)

            async with completed(
                    __name__, 'bbox_head_forward',
                    sleep_interval=sleep_interval):
                cls_score, bbox_pred = self.bbox_head(roi_feats)

            img_shape = img_metas[0]['img_shape']
            scale_factor = img_metas[0]['scale_factor']
            det_bboxes, det_labels = self.bbox_head.get_bboxes(
                rois,
                cls_score,
                bbox_pred,
                img_shape,
                scale_factor,
                rescale=rescale,
                cfg=rcnn_test_cfg)
            return det_bboxes, det_labels

    def test_zsd(self, rcnn_test_cfg, cls_score, bbox_pred):
        unseen_class_inds = get_asd_zero_shot_class_ids(dataset=rcnn_test_cfg.dataset_name, split=rcnn_test_cfg.split, fs_det=3)
        # seen_class_inds = get_seen_class_ids(dataset=rcnn_test_cfg.dataset_name, split=rcnn_test_cfg.split)
        seen_class_inds = np.array(list(set(list(range(80))) - set(unseen_class_inds)))
        
        seen_score = cls_score[:, seen_class_inds]
        unseen_score = cls_score[:, unseen_class_inds]

        seen=seen_score.argmax(1)
        unseen=unseen_score.argmax(1)
        seen = seen_class_inds[seen.data.cpu().numpy()]
        unseen = unseen_class_inds[unseen.data.cpu().numpy()]

        ar = torch.from_numpy(np.array([0,1,2,3]))

        for ii in range(seen.shape[0]):
            bbox_pred[ii, ar+(unseen[ii]*4) ] = bbox_pred[ii, ar+(seen[ii]*4) ]
        cls_score[:,seen_class_inds]=-1.0e3
        cls_score = F.softmax(cls_score, dim=1)

        return cls_score, bbox_pred

    def test_asd(self, rcnn_test_cfg, cls_score, bbox_pred):
        
        zero_shot_class_inds = get_asd_zero_shot_class_ids(dataset=rcnn_test_cfg.dataset_name, split=rcnn_test_cfg.split, fs_set=3)
        # seen_class_inds = get_seen_class_ids(dataset=rcnn_test_cfg.dataset_name, split=rcnn_test_cfg.split)
        
        base_class_inds = np.array([ 0,  1,  2,  3,  5,  7,  8,  9, 10, 11, 13, 14, 16, 17, 18, 19, 20,
        22, 23, 24, 25, 26, 27, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
        43, 44, 45, 46, 47, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 62,
        63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 79])
        base_and_few_class_inds = np.array(list(set(list(range(80))) - set(zero_shot_class_inds)))

        base_and_few_class_inds_bg = np.concatenate((base_and_few_class_inds, [81]))
        zero_shot_class_inds_bg = np.concatenate((zero_shot_class_inds, [80]))
        
        cls_score[:, base_and_few_class_inds_bg] = F.softmax(cls_score[:, base_and_few_class_inds_bg], dim=1)
        cls_score[:, zero_shot_class_inds_bg] = F.softmax(cls_score[:, zero_shot_class_inds_bg], dim=1)
        
        base_and_few_score = cls_score[:, base_and_few_class_inds]
        zero_shot_score = cls_score[:, zero_shot_class_inds]

        base_and_few = base_and_few_score.argmax(1)
        zero_shot = zero_shot_score.argmax(1)
        base_and_few = base_and_few_class_inds[base_and_few.data.cpu().numpy()]
        zero_shot = zero_shot_class_inds[zero_shot.data.cpu().numpy()]

        ar = torch.from_numpy(np.array([0,1,2,3]))
        base_and_few = base_and_few*4
        base_and_few = base_and_few[:, np.newaxis]
        base_and_few = np.repeat(base_and_few, 4, axis=1)
        zero_shot = zero_shot*4
        zero_shot = zero_shot[:, np.newaxis]
        zero_shot = np.repeat(zero_shot, 4, axis=1)
        ar = ar[np.newaxis]
        ar = np.repeat(ar, zero_shot.shape[0], axis=0)
        ar = ar.numpy()
        zero_shot_loc = zero_shot+ar
        base_and_few_loc = base_and_few+ar
        base_and_few_loc = torch.from_numpy(base_and_few_loc).cuda()
        zero_shot_loc = torch.from_numpy(zero_shot_loc).cuda()
        tmp_zero_shot = torch.gather(bbox_pred, 1, zero_shot_loc)
        tmp_base_and_few = torch.gather(bbox_pred, 1, base_and_few_loc)
        bbox_pred.scatter_(1, zero_shot_loc, tmp_base_and_few)
        # for ii in range(base_and_few.shape[0]):
        #     bbox_pred[ii, ar+(zero_shot[ii]*4) ] = bbox_pred[ii, ar+(base_and_few[ii]*4) ]
        cls_score[:,base_class_inds]=-1.0e3
        new_scores = cls_score[:, :-1]
        new_scores[:, -1] = (cls_score[:, -2] + cls_score[:, -1]) / 2
        cls_score = new_scores
        # cls_score = F.softmax(cls_score, dim=1)

        return cls_score, bbox_pred

    # merge 73 base+few-shot and 7 zero-shot classes
    def test_gasd(self, rcnn_test_cfg, cls_score, bbox_pred):
        # t0 = time.time()
        unseen_class_inds = get_asd_zero_shot_class_ids(dataset=rcnn_test_cfg.dataset_name, split=rcnn_test_cfg.split, fs_set=rcnn_test_cfg.fs_set)
        seen_class_inds = np.array(list(set(list(range(80))) - set(unseen_class_inds)))
        
        ## for quick eval, reduce function calls
        # unseen_class_inds = np.array([ 4, 28, 29, 48, 52, 61, 78])
        # seen_class_inds = np.array([ 0,  1,  2,  3,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
        #                     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36,
        #                     37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 50, 51, 53, 54, 55,
        #                     56, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73,
        #                     74, 75, 76, 77, 79])
        
        seen_class_inds_bg = np.concatenate((seen_class_inds, [81]))
        unseen_class_inds_bg = np.concatenate((unseen_class_inds, [80]))
        # temperature_factor = 0.25
        cls_score[:, seen_class_inds_bg] = F.softmax(cls_score[:, seen_class_inds_bg], dim=1)
        cls_score[:, unseen_class_inds_bg] = F.softmax(cls_score[:, unseen_class_inds_bg], dim=1)
        
        seen_score = cls_score[:, seen_class_inds]
        unseen_score = cls_score[:, unseen_class_inds] 

        ## 测试零样本背景类分类器和seen的背景类分类器的差异
        # _seen_bg_score = cls_score[:, seen_class_inds_bg]
        # _unseen_bg_score = cls_score[:, unseen_class_inds_bg]
        # _seen = _seen_bg_score.argmax(1)
        # _unseen = _unseen_bg_score.argmax(1)
        # num = _unseen[_unseen!=7].shape[0]
        # _unseen = _unseen.cpu().numpy()
        # if num > 0:
        #     print(cls_score[np.ix_(_unseen!=7, unseen_class_inds)])
        #     ipdb.set_trace()

        
        seen = seen_score.argmax(1)
        unseen = unseen_score.argmax(1)
        # ipdb.set_trace()
        seen = seen_class_inds[seen.data.cpu().numpy()] # get seen label index order in original coco 80 label index order 
        unseen = unseen_class_inds[unseen.data.cpu().numpy()]
        ar = torch.from_numpy(np.array([0,1,2,3]))
        # 将预测概率最大的seen label所对应的bbox预测结果复制给unseen预测概率值最大的bbox预测结果
        # t1 = time.time()
        seen = seen*4
        seen = seen[:, np.newaxis]
        seen = np.repeat(seen, 4, axis=1)
        unseen = unseen*4
        unseen = unseen[:, np.newaxis]
        unseen = np.repeat(unseen, 4, axis=1)
        ar = ar[np.newaxis]
        ar = np.repeat(ar, unseen.shape[0], axis=0)
        ar = ar.numpy()
        unseen_loc = unseen+ar
        seen_loc = seen+ar
        seen_loc = torch.from_numpy(seen_loc).cuda()
        unseen_loc = torch.from_numpy(unseen_loc).cuda()
        tmp_unseen = torch.gather(bbox_pred, 1, unseen_loc)
        tmp_seen = torch.gather(bbox_pred, 1, seen_loc)
        bbox_pred.scatter_(1, unseen_loc, tmp_seen)
        
        # t2 = time.time()
        
        # for ii in range(bbox_pred.shape[0]):
        #     bbox_pred[ii, ar+(unseen[ii]*4) ] = bbox_pred[ii, ar+(seen[ii]*4) ]
            # cls_rel_count_matrix[unseen[ii], seen[ii]] += 1
            # cls_rel_count_matrix[seen[ii], unseen[ii]] += 1
        # seen_score[seen_score < 0.3] = 0.0
        
        cls_score[:, seen_class_inds] = seen_score
        cls_score[:, unseen_class_inds] = unseen_score

        new_scores = cls_score[:, :-1]
        # new_scores[:, -1] = (cls_score[:, -2] + cls_score[:, -1]) / 2
        new_scores[:, -1] = cls_score[:, -1]
        cls_score = new_scores
        # t3 = time.time()
        
        return cls_score, bbox_pred
    
    # merge 73 base+few-shot and 15 zero-shot classes
    def test_gasd_v2(self, rcnn_test_cfg, cls_score, bbox_pred):
        few_shot_class_inds = np.array([4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78])
        base_class_inds = np.array([0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 79])
        zero_shot_class_inds = get_asd_zero_shot_class_ids(dataset=rcnn_test_cfg.dataset_name, split=rcnn_test_cfg.split, fs_set=rcnn_test_cfg.fs_set)
        base_and_few_shot_class_inds = np.array(list(set(list(range(80))) - set(zero_shot_class_inds)))

        base_and_few_shot_class_inds_bg = np.concatenate((base_and_few_shot_class_inds, [81]))
        additional_few_shot_class_inds = np.array(list(range(82, 82+15)))
        additional_few_shot_class_inds_bg = np.concatenate((list(range(82, 82+15)), [80]))
        zero_shot_class_inds_bg = np.concatenate((zero_shot_class_inds, [80]))

        cls_score[:, base_and_few_shot_class_inds_bg] = F.softmax(cls_score[:, base_and_few_shot_class_inds_bg], dim=1)
        cls_score[:, additional_few_shot_class_inds_bg] = F.softmax(cls_score[:, additional_few_shot_class_inds_bg], dim=1)
        cls_score[:, zero_shot_class_inds] = F.softmax(cls_score[:, zero_shot_class_inds_bg], dim=1)[:, :-1]

        base_and_few_shot_class_score = cls_score[:, base_and_few_shot_class_inds]
        additional_few_shot_class_score = cls_score[:, additional_few_shot_class_inds]
        zero_shot_class_score = cls_score[:, zero_shot_class_inds]

        base_and_few_shot = base_and_few_shot_class_score.argmax(1)
        additional_few_shot = additional_few_shot_class_score.argmax(1)
        zero_shot = zero_shot_class_score.argmax(1)
        
        base_and_few_shot = base_and_few_shot_class_inds[base_and_few_shot.data.cpu().numpy()] 
        additional_few_shot = additional_few_shot_class_inds[additional_few_shot.data.cpu().numpy()]
        zero_shot = zero_shot_class_inds[zero_shot.data.cpu().numpy()]

        ## judge if the additional few-shot class is in the zero_shot_class_inds
        flag = additional_few_shot in zero_shot_class_inds
        cls_score[flag, zero_shot_class_inds] = 0.0

        zero_shot_in_few_shot_inds = np.where(np.isin(few_shot_class_inds, zero_shot_class_inds))[0]
        cls_score[not flag, zero_shot_class_inds] = additional_few_shot_class_score[not flag, zero_shot_in_few_shot_inds]

        ar = torch.from_numpy(np.array([0,1,2,3]))
        base_and_few_shot = base_and_few_shot * 4
        base_and_few_shot = base_and_few_shot[:, np.newaxis]
        base_and_few_shot = np.repeat(base_and_few_shot, 4, axis=1)
        zero_shot = zero_shot * 4
        zero_shot = zero_shot[:, np.newaxis]
        zero_shot = np.repeat(zero_shot, 4, axis=1)
        ar = ar[np.newaxis]
        ar = np.repeat(ar, zero_shot.shape[0], axis=0)
        ar = ar.numpy()
        zero_shot_loc = zero_shot + ar
        base_and_few_shot_loc = base_and_few_shot + ar
        base_and_few_shot_loc = torch.from_numpy(base_and_few_shot_loc).cuda()
        zero_shot_loc = torch.from_numpy(zero_shot_loc).cuda()
        tmp_zero_shot = torch.gather(bbox_pred, 1, zero_shot_loc)
        tmp_base_and_few_shot = torch.gather(bbox_pred, 1, base_and_few_shot_loc)
        bbox_pred.scatter_(1, zero_shot_loc, tmp_base_and_few_shot)

        cls_score[:, base_and_few_shot_class_inds] = base_and_few_shot_class_score
        cls_score[:, zero_shot_class_inds] = zero_shot_class_score

        new_scores = cls_score[:, :81]
        new_scores[:, -1] = cls_score[:, 81]
        cls_score = new_scores

        return cls_score, bbox_pred


    def test_fsd(self, rcnn_test_cfg, cls_score, bbox_pred):
        few_shot_class_inds = np.array([4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78])
        base_class_inds = np.array([0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 79])
    
        base_score = cls_score[:, base_class_inds]
        few_shot_score = cls_score[:, few_shot_class_inds]
        base_argmax_id = base_score.argmax(1)
        few_shot_argmax_id = few_shot_score.argmax(1)
        base = base_class_inds[base_argmax_id.data.cpu().numpy()]
        few_shot = few_shot_class_inds[few_shot_argmax_id.data.cpu().numpy()]
        ar = torch.from_numpy(np.array([0,1,2,3]))
        # ipdb.set_trace()
        for ii in range(base.shape[0]):
            bbox_pred[ii, ar+(few_shot[ii]*4) ] = bbox_pred[ii, ar+(base[ii]*4) ]
        cls_score[:, base_class_inds] = -1e3
        cls_score = F.softmax(cls_score, dim=1)
        return cls_score, bbox_pred

    def test_gfsd(self, rcnn_test_cfg, cls_score, bbox_pred):
        few_shot_class_inds = np.array([4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78])
        base_class_inds = np.array([0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 79])
        base_class_inds_bg = np.concatenate((base_class_inds, [81]))
        few_shot_class_inds_bg = np.concatenate((few_shot_class_inds, [80]))
        cls_score[:, base_class_inds_bg] = F.softmax(cls_score[:, base_class_inds_bg], dim=1)
        cls_score[:, few_shot_class_inds_bg] = F.softmax(cls_score[:, few_shot_class_inds_bg], dim=1)

        
        base_score = cls_score[:, base_class_inds]
        few_shot_score = cls_score[:, few_shot_class_inds]

        
        base_argmax_id = base_score.argmax(1)
        few_shot_argmax_id = few_shot_score.argmax(1)
        base = base_class_inds[base_argmax_id.data.cpu().numpy()]
        few_shot = few_shot_class_inds[few_shot_argmax_id.data.cpu().numpy()]
        ar = torch.from_numpy(np.array([0,1,2,3]))
        # ipdb.set_trace()
        for ii in range(bbox_pred.shape[0]):
            bbox_pred[ii, ar+(few_shot[ii]*4) ] = bbox_pred[ii, ar+(base[ii]*4) ]
        
        base_score[base_score < 0.3] = 0.0
        cls_score[:, base_class_inds] = base_score
        cls_score[:, few_shot_class_inds] = few_shot_score
        
        # new_scores = cls_score[:, :-1]
        # new_scores[:, -1] = (cls_score[:, -2] + cls_score[:, -1]) / 2
        # cls_score = new_scores
        # ipdb.set_trace()
        cls_score = cls_score[:, :-1]
        
        return cls_score, bbox_pred

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.
        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """
        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois) 
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        if self.bbox_head.gasd:#rcnn_test_cfg.gfsd:
            cls_score, bbox_pred = self.test_gasd(rcnn_test_cfg, cls_score, bbox_pred)
            # cls_score, bbox_pred = self.test_gasd_v2(rcnn_test_cfg, cls_score, bbox_pred)
        elif self.bbox_head.asd: # rcnn_test_cfg.fsd:
            cls_score, bbox_pred = self.test_asd(rcnn_test_cfg, cls_score, bbox_pred)
        else:
            cls_score = F.softmax(cls_score, dim=1)

        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)
        
        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    do_softmax=False,
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        
        return det_bboxes, det_labels

    def aug_test_bboxes(self, feats, img_metas, proposal_list, rcnn_test_cfg):
        """Test det bboxes with test time augmentation."""
        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            img_shape = img_meta[0]['img_shape']
            scale_factor = img_meta[0]['scale_factor']
            flip = img_meta[0]['flip']
            flip_direction = img_meta[0]['flip_direction']
            # TODO more flexible
            proposals = bbox_mapping(proposal_list[0][:, :4], img_shape,
                                     scale_factor, flip, flip_direction)
            rois = bbox2roi([proposals])
            bbox_results = self._bbox_forward(x, rois)
            bboxes, scores = self.bbox_head.get_bboxes(
                rois,
                bbox_results['cls_score'],
                bbox_results['bbox_pred'],
                img_shape,
                scale_factor,
                rescale=False,
                cfg=None)
            aug_bboxes.append(bboxes)
            aug_scores.append(scores)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas, rcnn_test_cfg)
        if merged_bboxes.shape[0] == 0:
            # There is no proposal in the single image
            det_bboxes = merged_bboxes.new_zeros(0, 5)
            det_labels = merged_bboxes.new_zeros((0, ), dtype=torch.long)
        else:
            det_bboxes, det_labels = multiclass_nms(merged_bboxes,
                                                    merged_scores,
                                                    rcnn_test_cfg.score_thr,
                                                    rcnn_test_cfg.nms,
                                                    rcnn_test_cfg.max_per_img)
        return det_bboxes, det_labels


class MaskTestMixin_visual_info_transfer_73_7_two_softmax_from_suzsd:

    if sys.version_info >= (3, 7):

        async def async_test_mask(self,
                                  x,
                                  img_metas,
                                  det_bboxes,
                                  det_labels,
                                  rescale=False,
                                  mask_test_cfg=None):
            """Asynchronized test for mask head without augmentation."""
            # image shape of the first image in the batch (only one)
            ori_shape = img_metas[0]['ori_shape']
            scale_factor = img_metas[0]['scale_factor']
            if det_bboxes.shape[0] == 0:
                segm_result = [[] for _ in range(self.mask_head.num_classes)]
            else:
                if rescale and not isinstance(scale_factor,
                                              (float, torch.Tensor)):
                    scale_factor = det_bboxes.new_tensor(scale_factor)
                _bboxes = (
                    det_bboxes[:, :4] *
                    scale_factor if rescale else det_bboxes)
                mask_rois = bbox2roi([_bboxes])
                mask_feats = self.mask_roi_extractor(
                    x[:len(self.mask_roi_extractor.featmap_strides)],
                    mask_rois)

                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
                if mask_test_cfg and mask_test_cfg.get('async_sleep_interval'):
                    sleep_interval = mask_test_cfg['async_sleep_interval']
                else:
                    sleep_interval = 0.035
                async with completed(
                        __name__,
                        'mask_head_forward',
                        sleep_interval=sleep_interval):
                    mask_pred = self.mask_head(mask_feats)
                segm_result = self.mask_head.get_seg_masks(
                    mask_pred, _bboxes, det_labels, self.test_cfg, ori_shape,
                    scale_factor, rescale)
            return segm_result

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Simple test for mask head without augmentation."""
        # image shapes of images in the batch
        ori_shapes = tuple(meta['ori_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        if isinstance(scale_factors[0], float):
            warnings.warn(
                'Scale factor in img_metas should be a '
                'ndarray with shape (4,) '
                'arrange as (factor_w, factor_h, factor_w, factor_h), '
                'The scale_factor with float type has been deprecated. ')
            scale_factors = np.array([scale_factors] * 4, dtype=np.float32)

        num_imgs = len(det_bboxes)
        if all(det_bbox.shape[0] == 0 for det_bbox in det_bboxes):
            segm_results = [[[] for _ in range(self.mask_head.num_classes)]
                            for _ in range(num_imgs)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale:
                scale_factors = [
                    torch.from_numpy(scale_factor).to(det_bboxes[0].device)
                    for scale_factor in scale_factors
                ]
            _bboxes = [
                det_bboxes[i][:, :4] *
                scale_factors[i] if rescale else det_bboxes[i][:, :4]
                for i in range(len(det_bboxes))
            ]
            mask_rois = bbox2roi(_bboxes)
            mask_results = self._mask_forward(x, mask_rois)
            mask_pred = mask_results['mask_pred']
            # split batch mask prediction back to each image
            num_mask_roi_per_img = [len(det_bbox) for det_bbox in det_bboxes]
            mask_preds = mask_pred.split(num_mask_roi_per_img, 0)

            # apply mask post-processing to each image individually
            segm_results = []
            for i in range(num_imgs):
                if det_bboxes[i].shape[0] == 0:
                    segm_results.append(
                        [[] for _ in range(self.mask_head.num_classes)])
                else:
                    segm_result = self.mask_head.get_seg_masks(
                        mask_preds[i], _bboxes[i], det_labels[i],
                        self.test_cfg, ori_shapes[i], scale_factors[i],
                        rescale)
                    segm_results.append(segm_result)
        return segm_results

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        """Test for mask head with test time augmentation."""
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                flip_direction = img_meta[0]['flip_direction']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip, flip_direction)
                mask_rois = bbox2roi([_bboxes])
                mask_results = self._mask_forward(x, mask_rois)
                # convert to numpy array to save memory
                aug_masks.append(
                    mask_results['mask_pred'].sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas, self.test_cfg)

            ori_shape = img_metas[0][0]['ori_shape']
            scale_factor = det_bboxes.new_ones(4)
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg,
                ori_shape,
                scale_factor=scale_factor,
                rescale=False)
        return segm_result
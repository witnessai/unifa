# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from .bbox_head_fsd import BBoxHead_fsd
import ipdb 
import torch

import json
import torch.nn.functional as F

@HEADS.register_module()
class ConvFCBBoxHead_with_dpif_add_asso_loss_inference_fusion(BBoxHead_fsd):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.
    .. code-block:: none
                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead_with_dpif_add_asso_loss_inference_fusion, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_cls_convs +
                num_cls_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_cls_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_cls:
            assert num_cls_convs == 0 and num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch
        self.cls_convs, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        if init_cfg is None:
            # when init_cfg is None,
            # It has been set to
            # [[dict(type='Normal', std=0.01, override=dict(name='fc_cls'))],
            #  [dict(type='Normal', std=0.001, override=dict(name='fc_reg'))]
            # after `super(ConvFCBBoxHead, self).__init__()`
            # we only need to append additional configuration
            # for `shared_fcs`, `cls_fcs` and `reg_fcs`
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='shared_fcs'),
                        dict(name='cls_fcs'),
                        dict(name='reg_fcs')
                    ])
            ]
        self.gasd = False
        self.asd = False
        self.seen_bg_weight = None
        self.seen_bg_bias = None
        self.fc_asso = nn.Linear(self.cls_last_dim, 20)
        # load cls2asso_distribution
        cls2asso_distribution_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/data/few_shot_ann_60_20_split/coco/cls2asso_coco6020_w2v_dist_5.json'
        with open(cls2asso_distribution_path) as f:
            cls2asso = json.load(f)
        cats = ['truck', 'traffic light', 'fire hydrant', 'stop sign',
                  'parking meter', 'bench', 'elephant', 'bear', 'zebra',
                  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                  'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                  'kite', 'baseball bat', 'baseball glove', 'skateboard',
                  'surfboard', 'tennis racket', 'wine glass', 'cup', 'fork',
                  'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                  'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                  'cake', 'bed', 'toilet', 'laptop', 'mouse', 'remote',
                  'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                  'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                  'teddy bear', 'hair drier', 'toothbrush']
        unseen_cats = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                   'train', 'boat', 'bird', 'cat', 'dog', 'horse', 'sheep',
                   'cow', 'bottle', 'chair', 'couch', 'potted plant',
                   'dining table', 'tv']
        bg_cats = ['__background__']
        all_cats = cats + unseen_cats + bg_cats
        id2asso_dist = torch.zeros((len(all_cats), len(unseen_cats)))
        for i, c in enumerate(all_cats):
            if c == '__background__':
                continue
            association = cls2asso[c]
            if type(association) is list:
                id2asso_dist[i] = torch.tensor(association)
            else:
                id2asso_dist[i][association] = 1
        # shape is (81, 20)
        self.id2asso_dists = id2asso_dist.cuda()
        #torch.transpose(id2asso_dist.cuda(), 0, 1)
    
    def asso_loss(self, asso_prob, cls_label):
        asso_dist = self.id2asso_dists[cls_label]
        # print(asso_prob.shape, asso_dist.shape)
        loss = F.binary_cross_entropy(asso_prob, asso_dist)
        return loss
        

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.
        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x, roi_labels=None):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x
        x_asso = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        asso_score = self.fc_asso(x_asso)
        asso_prob = torch.sigmoid(asso_score)
        # print(asso_prob)
        if self.training:
            asso_loss = self.asso_loss(asso_prob, roi_labels)
            asso_loss.backward(retain_graph=True)
        if not self.training:
            asso_prob = torch.mm(asso_prob, torch.transpose(self.id2asso_dists, 0, 1) )
            return [cls_score, asso_prob], bbox_pred
            # ipdb.set_trace()
            # cls_score = cls_score * asso_prob


        # if self.gasd or self.asd:
        #     y_seen_bg = torch.mm(x_cls, self.seen_bg_weight[:, None]) + self.seen_bg_bias
        #     cls_score = torch.cat((cls_score, y_seen_bg), dim=1)

        # import torch
        # prob = torch.softmax(cls_score, 1)
        # values, indices = torch.max(prob, 1)
        # zero_shot_label_id = [4, 15, 28, 29, 48, 61, 64]
        # if torch.max(prob[:, zero_shot_label_id]) > 0.01:
        #     max_val = torch.max(prob[:, zero_shot_label_id])
        #     print(max_val)
        #     for i in range(1000):
        #         for j in zero_shot_label_id:
        #             if prob[i, j] > max_val-0.001 and prob[i, j] < max_val+0.001:
        #                 print(i, j)
        #                 break
        #     print('finish')
        # for label in zero_shot_label_id:
        #     if label in indices:
        #         ipdb.set_trace()
        return cls_score, bbox_pred


@HEADS.register_module()
class Shared2FCBBoxHead_with_dpif_add_asso_loss_inference_fusion(ConvFCBBoxHead_with_dpif_add_asso_loss_inference_fusion):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHead_with_dpif_add_asso_loss_inference_fusion, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=2,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class Shared4Conv1FCBBoxHead_with_dpif_add_asso_loss_inference_fusion(ConvFCBBoxHead_with_dpif_add_asso_loss_inference_fusion):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxHead_with_dpif_add_asso_loss_inference_fusion, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
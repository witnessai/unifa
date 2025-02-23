# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from .bbox_head_visual_info_transfer_attention_generator import BBoxHead_visual_info_transfer_attention_generator
import ipdb 
import torch
import torch.nn.functional as F 

@HEADS.register_module()
class ConvFCBBoxHead_visual_info_transfer_attention_generator(BBoxHead_visual_info_transfer_attention_generator):
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
        super(ConvFCBBoxHead_visual_info_transfer_attention_generator, self).__init__(
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
        
        # load fs classifier trained for generating features
        model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_65_15/0.6725_best_acc_in_testdata_20230208_text_embedding/classifier_best.pth'
        checkpoint = torch.load(model_path)
        tar_size = checkpoint['state_dict']['fc1.weight'].shape
        self.gen_fs_classifier = nn.Linear(tar_size[1], tar_size[0])
        for i in range(tar_size[0]):
            self.gen_fs_classifier.weight[i] = checkpoint['state_dict']['fc1.weight'][i]
            self.gen_fs_classifier.bias[i] = checkpoint['state_dict']['fc1.bias'][i]
        for param in self.gen_fs_classifier.parameters():
                param.detach_()
        # self.gen_fs_classifier.weight.requires_grad = False
        # self.gen_fs_classifier.bias.requires_grad = False
        # self.attention_generator_v1 = nn.Linear(tar_size[0], tar_size[0])
        self.attention_generator_v2 = nn.Linear(tar_size[0]+self.num_classes+1, tar_size[0])
        self.fs_atten = None

    
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

    def forward(self, x):
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

        fs_score = self.gen_fs_classifier(x_cls) # few-shot score
        COCO_FS_CLASSES = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78]
        COCO_FS_CLASSES_AND_BG = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78, 80]
        
        # (1) 融合fs 15类和背景类
        # fs_atten = self.attention_generator_v1(fs_score)
        # fs_atten_relu = F.relu(fs_atten )
        # cls_score[:, COCO_FS_CLASSES_AND_BG] = cls_score[:, COCO_FS_CLASSES_AND_BG] * fs_atten_relu[:, :len(COCO_FS_CLASSES_AND_BG)]
        # (2) 只融合fs 15类
        # fs_atten = self.attention_generator_v1(fs_score)
        # fs_atten_relu = F.relu(fs_atten )
        # cls_score[:, COCO_FS_CLASSES] = cls_score[:, COCO_FS_CLASSES] * fs_atten_relu[:, :len(COCO_FS_CLASSES)]
        # (3) attention generator的输入改为cls_score+fs_score
        # fs_atten = self.attention_generator_v2(torch.cat([cls_score, fs_score], 1))
        # cls_score[:, COCO_FS_CLASSES] = fs_atten[:, :len(COCO_FS_CLASSES)]
        # (4) 在(3)的基础上增加输出logits最大值分布的约束
        # fs_atten = self.attention_generator_v2(torch.cat([cls_score, fs_score], 1))
        
        # _, ind_cls = torch.max(cls_score[:, COCO_FS_CLASSES], 1)
        # _, ind_fss = torch.max(fs_score[:, :len(COCO_FS_CLASSES)], 1)
        # _, ind_fsa = torch.max(fs_atten[:, :len(COCO_FS_CLASSES)], 1)
        
        # loss_max_loc_constrain = 1-torch.sum((ind_cls==ind_fsa) |(ind_fss==ind_fsa))/ind_cls.shape[0]
        # cls_score[:, COCO_FS_CLASSES] = fs_atten[:, :len(COCO_FS_CLASSES)]
        # return cls_score, bbox_pred, loss_max_loc_constrain
        # (5) attention generator的输入为cls_score+fs_score，输出为概率的scalar
        fs_atten = self.attention_generator_v2(torch.cat([cls_score, fs_score], 1))
        fs_atten = F.relu(fs_atten)
        self.fs_atten = fs_atten


        if self.gasd or self.asd:
            y_seen_bg = torch.mm(x_cls, self.seen_bg_weight[:, None]) + self.seen_bg_bias
            cls_score = torch.cat((cls_score, y_seen_bg), dim=1)
  
        return cls_score, bbox_pred, fs_atten
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
        


@HEADS.register_module()
class Shared2FCBBoxHead_visual_info_transfer_attention_generator(ConvFCBBoxHead_visual_info_transfer_attention_generator):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHead_visual_info_transfer_attention_generator, self).__init__(
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
class Shared4Conv1FCBBoxHead_visual_info_transfer_attention_generator(ConvFCBBoxHead_visual_info_transfer_attention_generator):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxHead_visual_info_transfer_attention_generator, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
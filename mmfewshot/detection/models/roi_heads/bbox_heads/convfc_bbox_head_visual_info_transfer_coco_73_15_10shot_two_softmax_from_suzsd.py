# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from .bbox_head_suzsd_get_bbox_without_softmax import BBoxHead_suzsd_get_bbox_without_softmax
from .bbox_head2 import BBoxHead2
import ipdb 
import torch 

import torch.nn.functional as F 
import numpy as np

@HEADS.register_module()
class ConvFCBBoxHead_visual_info_transfer_coco_73_15_10shot_two_softmax_from_suzsd(BBoxHead_suzsd_get_bbox_without_softmax):
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
                 asd=None, 
                 gasd=None, 
                 fs_set=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHead_visual_info_transfer_coco_73_15_10shot_two_softmax_from_suzsd, self).__init__(
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
        
        self.gfsd = False
        self.fsd = False
        
        # load fs classifier finetuned
        self.load_fs_flag = False
        self.gasd = gasd
        self.asd = asd
        self.fs_set = fs_set 
        self.seen_bg_weight = None
        self.seen_bg_bias = None
        self.load_bg_classifier = False
        self.bg_classifier = None
        self.use_suzsd_weighted_combine = True
        self.model_path = None
        self.only_use_zs_classifier = False # for 73 15 config, default is False
        self.additional_fs_classifier = nn.Linear(1024, 16).cuda()
        
        
    
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
        # ipdb.set_trace()
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
        
        if self.load_fs_flag is False:
            self.load_fs_flag = True # forward时只加载一次即可
            
            ## 7类分类器
            # self.model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/work_dirs/fsd_65_15_r101_fpn_coco_1shot-fine-tuning/classifier_best_finetuning_on_gen_feats_epoch4.pth'
            # self.model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/work_dirs/fsd_65_15_r101_fpn_coco_1shot-fine-tuning/from_scratch_train_classifier_1shot_on_gen_featsepoch15.pth'
            # self.model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_65_15/0.6725_best_acc_in_testdata_20230208_text_embedding/classifier_best_latest.pth'
            # self.model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_65_15/0.5639_best_acc_in_testdata_20230208_word_embedding_fasttext/classifier_best_latest.pth'

            ## zsdscr
            self.model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_73_7/fs_set3/0.6375_best_acc_in_testdata_20230403_text_embedding/classifier_best_latest.pth'
            self.model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_73_7/fs_set3/0.7159_best_acc_in_testdata_20230403_text_embedding/classifier_best_latest.pth'
            self.model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_73_7/fs_set3/0.7159_best_acc_in_testdata_20230403_text_embedding/classifier_latest.pth'
            self.model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_73_7/fs_set3/0.7159_best_acc_in_testdata_20230406_text_embedding/classifier_latest.pth'
            self.model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_73_7/fs_set3/0.7128_best_acc_in_testdata_20230407_text_embedding/0.9278_classifier_best_bg.pth'

            ## 65/15分类器
            self.model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_65_15/0.6725_best_acc_in_testdata_20230208_text_embedding/classifier_best_latest.pth' # 8.1
            self.model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_65_15/both_regressor_triplet/0.6712_best_acc_in_testdata_20230403_text_embedding/classifier_best_latest.pth' # 7.9
            self.model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_65_15/0.6879_best_acc_in_testdata_20230419_text_embedding/classifier_best_latest_0.6879.pth' # 7.9
            self.model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_65_15/both_regressor_triplet/0.6836_best_acc_in_testdata_20230419_text_embedding_addmixup/classifier_best_latest_0.6836.pth' # 8.6
            # self.model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_65_15/both_regressor_triplet/0.6839_best_acc_in_testdata_20230421_text_embedding_addmixup_increasefeatnum/classifier_best_latest_0.6839.pth' # 7.8
            COCO_NOVEL_CLASSES_AND_BG = []
            checkpoint = torch.load(self.model_path)
            
            if 'fs_set1' in self.model_path:
                pass
            elif 'fs_set2' in self.model_path:
                pass
            elif 'fs_set3' in self.model_path:
                COCO_ZERO_SHOT_CLASSES_AND_BG = [4, 28, 29, 48, 52, 61, 78, 80]
                COCO_NOVEL_CLASSES_AND_BG = COCO_ZERO_SHOT_CLASSES_AND_BG
            # 65/15
            elif 'fsd_65_15' in self.model_path and self.only_use_zs_classifier:
                COCO_ZERO_SHOT_CLASSES_AND_BG = [4, 28, 29, 48, 52, 61, 78, 80]
                COCO_FEW_SHOT_CLASSES_AND_BG = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78, 80]
                inds = np.where(np.isin(COCO_FEW_SHOT_CLASSES_AND_BG, COCO_ZERO_SHOT_CLASSES_AND_BG))[0]
                checkpoint['state_dict']['fc1.weight'] = checkpoint['state_dict']['fc1.weight'][inds, :]
                checkpoint['state_dict']['fc1.bias'] = checkpoint['state_dict']['fc1.bias'][inds]
                COCO_NOVEL_CLASSES_AND_BG = COCO_ZERO_SHOT_CLASSES_AND_BG
            elif 'fsd_65_15' in self.model_path and not self.only_use_zs_classifier:
                COCO_FEW_SHOT_CLASSES_AND_BG = [4, 6, 12, 15, 21, 28, 29, 31, 42, 48, 52, 61, 64, 70, 78, 80]
                COCO_NOVEL_CLASSES_AND_BG = COCO_FEW_SHOT_CLASSES_AND_BG
            COCO_IDMAP = {v: i for i, v in enumerate(COCO_NOVEL_CLASSES_AND_BG)}
            COCO_IDMAP_reverse = {i: v for i, v in enumerate(COCO_NOVEL_CLASSES_AND_BG)}
            is_weight_list = [True, False]
            tar_size = [len(COCO_NOVEL_CLASSES_AND_BG), 1024]
            tmp_weight = nn.Linear(tar_size[1], tar_size[0])
            if 'finetuning_on_gen_feats' in self.model_path:
                param_name = 'fc'
            else:
                param_name = 'fc1'
            for is_weight in is_weight_list:
                weight_name = param_name + ('.weight' if is_weight else '.bias')
                pretrained_weight = checkpoint['state_dict'][weight_name] # shape is [81, 1024] or [81]
                for idx, c in enumerate(COCO_NOVEL_CLASSES_AND_BG):
                    if is_weight:
                        tmp_weight.weight[idx] = pretrained_weight[idx]
                    else:
                        tmp_weight.bias[idx] = pretrained_weight[idx]
            weight = torch.tensor(tmp_weight.weight.clone().detach()).numpy()
            bias = torch.tensor(tmp_weight.bias.clone().detach()).numpy()
            # ipdb.set_trace()            
            for idx, c in enumerate(COCO_NOVEL_CLASSES_AND_BG[:-1]):
                self.additional_fs_classifier.weight[idx] = nn.Parameter(torch.from_numpy(weight[idx]).float())
                self.additional_fs_classifier.bias[idx] = nn.Parameter(torch.tensor(bias[idx]))
            
            if 'fsd_65_15' in self.model_path and not self.only_use_zs_classifier:
                COCO_ZERO_SHOT_CLASSES_AND_BG = [4, 28, 29, 48, 52, 61, 78, 80]
                inds = np.where(np.isin(COCO_NOVEL_CLASSES_AND_BG, COCO_ZERO_SHOT_CLASSES_AND_BG))[0]
                for idx, c in enumerate(COCO_ZERO_SHOT_CLASSES_AND_BG[:-1]):
                    self.fc_cls.weight[c] = nn.Parameter(torch.from_numpy(weight[inds[idx]]).float())
                    self.fc_cls.bias[c] = nn.Parameter(torch.tensor(bias[inds[idx]]))
            if self.use_suzsd_weighted_combine:
                # 合并方式一： Ours
                # alpha1 = 0.35
                # alpha2 = 1-alpha1
                # self.seen_bg_weight = alpha1*self.fc_cls.weight[-1]+alpha2*checkpoint['state_dict'][param_name+'.weight'][-1]
                # self.seen_bg_bias = alpha1*self.fc_cls.bias[-1]+checkpoint['state_dict'][param_name+'.bias'][-1]
                # 合并方式二：suzsd
                alpha1 = 0.35
                alpha2 = 1-alpha1
                self.seen_bg_weight = self.fc_cls.weight[-1].clone().detach()
                self.seen_bg_bias = self.fc_cls.bias[-1].clone().detach()
                self.fc_cls.weight[-1] = alpha1*self.fc_cls.weight[-1]+alpha2*checkpoint['state_dict'][param_name+'.weight'][-1]
                self.fc_cls.bias[-1] = alpha1*self.fc_cls.bias[-1]+alpha2*checkpoint['state_dict'][param_name+'.bias'][-1]
            else:
                self.seen_bg_weight = checkpoint['state_dict'][param_name+'.weight'][-1]
                self.seen_bg_bias = checkpoint['state_dict'][param_name+'.bias'][-1]
            
            if self.load_bg_classifier:
                # self.model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_73_7/fs_set3/0.7159_best_acc_in_testdata_20230404_text_embedding/0.9444_classifier_best_bg.pth'
                self.model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/checkpoints/fsd_73_7/fs_set3/0.7159_best_acc_in_testdata_20230403_text_embedding/classifier_latest.pth'
                checkpoint = torch.load(self.model_path)
                self.bg_classifier = nn.Linear(tar_size[1], tar_size[0])
                for is_weight in is_weight_list:
                    weight_name = param_name + ('.weight' if is_weight else '.bias')
                    pretrained_weight = checkpoint['state_dict'][weight_name] # shape is [81, 1024] or [81]
                    for idx, c in enumerate(COCO_NOVEL_CLASSES_AND_BG):
                        if is_weight:
                            self.bg_classifier.weight[idx] = pretrained_weight[idx]
                        else:
                            self.bg_classifier.bias[idx] = pretrained_weight[idx]
                self.bg_classifier = self.bg_classifier.cuda()
                alpha1 = 0.35
                alpha2 = 1-alpha1
                self.bg_classifier.weight[-1] = alpha1*self.seen_bg_weight+alpha2*self.bg_classifier.weight[-1]
                self.bg_classifier.bias[-1] = alpha1*self.seen_bg_bias+alpha2*self.bg_classifier.bias[-1]
        
        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        if self.load_bg_classifier:
    
            if 'fs_set1' in self.model_path:
                pass
            elif 'fs_set2' in self.model_path:
                pass
            elif 'fs_set3' in self.model_path:
                COCO_ZERO_SHOT_CLASSES_AND_BG = [4, 28, 29, 48, 52, 61, 78, 80]
                COCO_NOVEL_CLASSES_AND_BG = COCO_ZERO_SHOT_CLASSES_AND_BG
            bg_classifier_logits = self.bg_classifier(x_cls)
            bg_classifier_score = F.softmax(bg_classifier_logits, 1)
            novel_cls_score = F.softmax(cls_score[:, COCO_NOVEL_CLASSES_AND_BG], 1)
            bg_classifier_score_argmax = torch.argmax(bg_classifier_score, 1)
            # inds = (bg_classifier_score_argmax == 7)
            # cls_score[np.ix_(inds.cpu().numpy(), COCO_NOVEL_CLASSES_AND_BG)] = bg_classifier_logits[inds, :]
            
            ## v1 code
            inds = bg_classifier_score[:, -1] > novel_cls_score[:, -1]
            cls_score[np.ix_(inds.cpu().numpy(), COCO_NOVEL_CLASSES_AND_BG)] = bg_classifier_logits[inds, :]

        if self.gfsd:
            y_seen_bg = torch.mm(x_cls, self.seen_bg_weight[:, None]) + self.seen_bg_bias
            cls_score = torch.cat((cls_score, y_seen_bg), dim=1)

        if self.gasd or self.asd:
            y_seen_bg = torch.mm(x_cls, self.seen_bg_weight[:, None]) + self.seen_bg_bias
            cls_score = torch.cat((cls_score, y_seen_bg), dim=1)
            # ipdb.set_trace()
            # additional_fs_score = torch.mm(x_cls, self.additional_fs_classifier.weight[:, None]) + self.additional_fs_classifier.bias
            additional_fs_score = self.additional_fs_classifier(x_cls)
            cls_score = torch.cat((cls_score, additional_fs_score[:, :-1]), dim=1)

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
class Shared2FCBBoxHead_visual_info_transfer_coco_73_15_10shot_two_softmax_from_suzsd(ConvFCBBoxHead_visual_info_transfer_coco_73_15_10shot_two_softmax_from_suzsd):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCBBoxHead_visual_info_transfer_coco_73_15_10shot_two_softmax_from_suzsd, self).__init__(
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
class Shared4Conv1FCBBoxHead_visual_info_transfer_coco_73_15_10shot_two_softmax_from_suzsd(ConvFCBBoxHead_visual_info_transfer_coco_73_15_10shot_two_softmax_from_suzsd):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCBBoxHead_visual_info_transfer_coco_73_15_10shot_two_softmax_from_suzsd, self).__init__(
            num_shared_convs=4,
            num_shared_fcs=1,
            num_cls_convs=0,
            num_cls_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
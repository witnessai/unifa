# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.roi_heads import ConvFCBBoxHead
from torch import Tensor


@HEADS.register_module()
class ContrastiveBBoxHead_visual_info_transfer_voc_15_5_3shot_finetune_det_classifier_fsce_add_cpe(ConvFCBBoxHead):
    """BBoxHead for `FSCE <https://arxiv.org/abs/2103.05950>`_.

    Args:
        mlp_head_channels (int): Output channels of contrast branch
            mlp. Default: 128.
        with_weight_decay (bool): Whether to decay loss weight. Default: False.
        loss_contrast (dict): Config of contrast loss.
        scale (int): Scaling factor of `cls_score`. Default: 20.
        learnable_scale (bool): Learnable global scaling factor.
            Default: False.
        eps (float): Constant variable to avoid division by zero.
    """

    def __init__(self,
                 mlp_head_channels: int = 128,
                 with_weight_decay: bool = False,
                 loss_contrast: Dict = dict(
                     type='SupervisedContrastiveLoss',
                     temperature=0.1,
                     iou_threshold=0.5,
                     loss_weight=1.0,
                     reweight_type='none'),
                 scale: int = 20,
                 learnable_scale: bool = False,
                 eps: float = 1e-5,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # override the fc_cls in :obj:`ConvFCBBoxHead`
        if self.with_cls:
            self.fc_cls = nn.Linear(
                self.cls_last_dim, self.num_classes + 1, bias=False)

        # learnable global scaling factor
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1) * scale)
        else:
            self.scale = scale
        self.mlp_head_channels = mlp_head_channels
        self.with_weight_decay = with_weight_decay
        self.eps = eps
        # This will be updated by :class:`ContrastiveLossDecayHook`
        # in the training phase.
        self._decay_rate = 1.0
        self.gamma = 1
        self.contrastive_head = nn.Sequential(
            nn.Linear(self.fc_out_channels, self.fc_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.fc_out_channels, mlp_head_channels))
        self.contrast_loss = build_loss(copy.deepcopy(loss_contrast))
        self.load_fs_flag = False

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward function.

        Args:
            x (Tensor): Shape of (num_proposals, C, H, W).

        Returns:
            tuple:
                cls_score (Tensor): Cls scores, has shape
                    (num_proposals, num_classes).
                bbox_pred (Tensor): Box energies / deltas, has shape
                    (num_proposals, 4).
                contrast_feat (Tensor): Box features for contrast loss,
                    has shape (num_proposals, C).
        """
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
        x_contra = x
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
            self.load_fs_flag = True
            # ipdb.set_trace()
            model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/work_dirs/fsce_r101_fpn_contrastive-loss_voc-split1_3shot-fine-tuning/classifier_best_finetuning_on_gen_feats_epoch3.pth'
            checkpoint = torch.load(model_path)
            COCO_NOVEL_CLASSES_AND_BG = [15, 16, 17, 18, 19, 20]
            COCO_IDMAP = {v: i for i, v in enumerate(COCO_NOVEL_CLASSES_AND_BG)}
            COCO_IDMAP_reverse = {i: v for i, v in enumerate(COCO_NOVEL_CLASSES_AND_BG)}
            is_weight_list = [True]
            tar_size = [6, 1024]
            tmp_weight = nn.Linear(tar_size[1], tar_size[0])
            param_name = 'fc'
            for is_weight in is_weight_list:
                weight_name = param_name + ('.weight' if is_weight else '.bias')
                pretrained_weight = checkpoint['state_dict'][weight_name] # shape is [81, 1024] or [81]
                for idx, c in enumerate(COCO_NOVEL_CLASSES_AND_BG):
                    if is_weight:
                        tmp_weight.weight[idx] = pretrained_weight[idx]
                    else:
                        tmp_weight.bias[idx] = pretrained_weight[idx]
            weight = torch.tensor(tmp_weight.weight.clone().detach()).numpy()
            # bias = torch.tensor(tmp_weight.weight.clone().detach()).numpy()            
            for idx, c in enumerate(COCO_NOVEL_CLASSES_AND_BG[:-1]):
                # ipdb.set_trace()
                self.fc_cls.weight[c] = nn.Parameter(torch.from_numpy(weight[idx]).float())
                # self.fc_cls.bias[c] = nn.Parameter(torch.from_numpy(bias[idx]).float())
            # ipdb.set_trace()

        # reg branch
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        # cls branch
        if x_cls.dim() > 2:
            x_cls = torch.flatten(x_cls, start_dim=1)

        # normalize the input x along the `input_size` dimension
        x_norm = torch.norm(x_cls, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_cls_normalized = x_cls.div(x_norm + self.eps)
        # normalize weight
        with torch.no_grad():
            temp_norm = torch.norm(
                self.fc_cls.weight, p=2,
                dim=1).unsqueeze(1).expand_as(self.fc_cls.weight)
            self.fc_cls.weight.div_(temp_norm + self.eps)
        # calculate and scale cls_score
        cls_score = self.scale * self.fc_cls(
            x_cls_normalized) if self.with_cls else None

        # contrastive branch
        contrast_feat = self.contrastive_head(x_contra)
        contrast_feat = F.normalize(contrast_feat, dim=1)

        return cls_score, bbox_pred, contrast_feat

    def set_decay_rate(self, decay_rate: float) -> None:
        """Contrast loss weight decay hook will set the `decay_rate` according
        to iterations.

        Args:
            decay_rate (float): Decay rate for weight decay.
        """
        self._decay_rate = decay_rate

    @force_fp32(apply_to=('contrast_feat'))
    def loss_contrast(self,
                      contrast_feat: Tensor,
                      proposal_ious: Tensor,
                      labels: Tensor,
                      reduction_override: Optional[str] = None) -> Dict:
        """Loss for contract.

        Args:
            contrast_feat (tensor): BBox features with shape (N, C)
                used for contrast loss.
            proposal_ious (tensor): IoU between proposal and ground truth
                corresponding to each BBox features with shape (N).
            labels (tensor): Labels for each BBox features with shape (N).
            reduction_override (str | None): The reduction method used to
                override the original reduction method of the loss. Options
                are "none", "mean" and "sum". Default: None.

        Returns:
            Dict: The calculated loss.
        """

        losses = dict()
        if self.with_weight_decay:
            decay_rate = self._decay_rate
        else:
            decay_rate = None
        losses['loss_contrast'] = self.contrast_loss(
            contrast_feat,
            labels,
            proposal_ious,
            decay_rate=decay_rate,
            reduction_override=reduction_override)
        return losses

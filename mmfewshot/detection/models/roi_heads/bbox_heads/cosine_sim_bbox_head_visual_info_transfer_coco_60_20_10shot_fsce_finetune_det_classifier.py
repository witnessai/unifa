# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

import torch
import torch.nn as nn
from mmdet.models.builder import HEADS
from mmdet.models.roi_heads import ConvFCBBoxHead
from torch import Tensor
import ipdb

@HEADS.register_module()
class CosineSimBBoxHead_visual_info_transfer_coco_60_20_10shot_fsce_finetune_det_classifier(ConvFCBBoxHead):


    def __init__(self,
                 scale: int = 20,
                 learnable_scale: bool = False,
                 eps: float = 1e-5,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        if self.with_cls:
            self.fc_cls = nn.Linear(
                self.cls_last_dim, self.num_classes + 1, bias=False)

        # learnable global scaling factor
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1) * scale)
        else:
            self.scale = scale
        self.eps = eps
        
        self.load_fs_flag = False

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
       
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
        
        if self.load_fs_flag is False:
            self.load_fs_flag = True
            # ipdb.set_trace()
            model_path = '/home/nieh/morjio/projects/detection/any_shot_detection/mmfewshot/work_dirs/fsce_r101_fpn_coco_10shot-fine-tuning_34server/classifier_best_finetuning_on_gen_feats_epoch6.pth'
            checkpoint = torch.load(model_path)
            COCO_NOVEL_CLASSES_AND_BG = [0, 1, 2, 3, 4, 5, 6, 8, 14, 15, 16, 17, 18, 19, 39, 56, 57, 58, 60, 62, 80]
            COCO_IDMAP = {v: i for i, v in enumerate(COCO_NOVEL_CLASSES_AND_BG)}
            COCO_IDMAP_reverse = {i: v for i, v in enumerate(COCO_NOVEL_CLASSES_AND_BG)}
            is_weight_list = [True]
            tar_size = [len(COCO_NOVEL_CLASSES_AND_BG), 1024]
            tmp_weight = nn.Linear(tar_size[1], tar_size[0])
            param_name = 'fc'
            for is_weight in is_weight_list:
                weight_name = param_name + ('.weight' if is_weight else '.bias')
                if weight_name not in checkpoint['state_dict'].keys():
                    continue 
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

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

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

        return cls_score, bbox_pred

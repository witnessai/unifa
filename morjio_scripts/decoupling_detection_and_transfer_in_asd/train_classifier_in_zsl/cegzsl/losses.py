from __future__ import print_function
import torch
import torch.nn as nn
import ipdb 

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.01):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - mean_log_prob_pos
        loss = loss.mean()
        return loss

# clear those instances that have no positive instances to avoid training error
class SupConLoss_clear(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss_clear, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] # features.shape is [2048, 512]
        labels = labels.contiguous().view(-1, 1) # labels.shape is [2048, 1]
        mask = torch.eq(labels, labels.T).float().to(device) # mask.shape is [2048, 2048]
        
        # 计算embedding的相似度，除以温度系数变为logits
        anchor_dot_contrast = torch.div(  # anchor_dot_contrast.shape is [2048, 2048]
            torch.matmul(features, features.T),
            self.temperature) # self.temperature is 0.1

        # normalize the logits for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) # logits_max.shape is [2048, 1], in which max value is 10.00
        logits = anchor_dot_contrast - logits_max.detach()

        # 构造全1矩阵，并将对角线的值变为0
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        # 构造mask，即除去自身以外的同标签位置标记为1，其他位置为0
        mask = mask * logits_mask
        single_samples = (mask.sum(1) == 0).float()
        # ipdb.set_trace()
        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # invoid to devide the zero
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1)+single_samples)

        # loss
        # filter those single sample
        loss = - mean_log_prob_pos*(1-single_samples)
        loss = loss.sum()/(loss.shape[0]-single_samples.sum())

        return loss
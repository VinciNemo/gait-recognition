# 监督对比损失 SupConLoss
# 参考: Supervised Contrastive Learning (Khosla et al., NeurIPS 2020)
# 把同一个人的不同视图拉近，把不同人推远，GaitGraph 训练用的就是这个

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        # features: (N, num_views, feat_dim)，通常已经 L2 归一化
        device = features.device
        if len(features.shape) < 3:
            raise ValueError("features 需要为 (N, num_views, feat_dim) 形状")
        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("labels 和 mask 不能同时提供")
        if labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)

        # 展平视图: (N*V, feat)
        features = features.contiguous().view(batch_size, -1, features.shape[-1])
        features = F.normalize(features, dim=2, p=2)

        # 计算全对比相似度矩阵
        # 将每个样本的视图与其他样本的所有视图做点积
        contrast_feature = features.reshape(-1, features.shape[-1])  # (N*V, feat)
        anchor_feature = contrast_feature
        anchor_count = contrast_feature.shape[0]

        # 相似度矩阵 (N*V, N*V)
        logits = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature,
        )

        # 构建正负样本 mask
        if labels is not None:
            mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
        # 扩展视图维度: 每个样本的所有视图之间互为对比
        mask = mask.repeat(anchor_count // batch_size, anchor_count // batch_size)

        # 排除自身
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(anchor_count).view(-1, 1).to(device), 0,
        )
        mask = mask * logits_mask

        # 对数项
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # 正样本对数平均
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        # 每样本损失，然后取平均
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, -1).mean(1).mean(0)
        loss = loss * self.base_temperature
        return loss

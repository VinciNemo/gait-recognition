# 精简版 OpenGait —— ResGCN 骨干网络
# 对应 OpenGait 的 backbones/resgcn.py，GaitGraph1 的主干：
#   输入分支（多流） + 主分支（时空图卷积 + 可学习边权重） + 分类头

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules import (SpatialBasicBlock, SpatialBottleneckBlock,
                       TemporalBasicBlock, TemporalBottleneckBlock)


class ResGCNModule(nn.Module):
    # 时空图卷积模块：空间块 + 时间块
    # A 是邻接矩阵 (K, V, V)；kernel_size=[时间核, 空间核]，通常 [9, 2]

    def __init__(self, in_channels, out_channels, block, A, stride=1,
                 kernel_size=(9, 2), reduction=4):
        super().__init__()
        temporal_window_size = kernel_size[0]
        max_graph_distance = kernel_size[1]

        if block == "Basic":
            spatial_block = SpatialBasicBlock
            temporal_block = TemporalBasicBlock
        elif block == "Bottleneck":
            spatial_block = SpatialBottleneckBlock
            temporal_block = TemporalBottleneckBlock
        else:
            raise ValueError(f"未知 block 类型: {block}")

        self.scn = spatial_block(in_channels, out_channels, max_graph_distance, reduction)
        self.tcn = temporal_block(out_channels, out_channels, temporal_window_size,
                                  stride, reduction=reduction)
        self.relu = nn.ReLU(inplace=True)
        # 可学习的边权重（OpenGait 中每个模块一个，作用于邻接矩阵）
        self.edge = nn.Parameter(torch.ones_like(A))

    def forward(self, x, A=None):
        if A is None:
            A = self.edge
        else:
            A = A * self.edge
        x = self.scn(x, A)
        x = self.tcn(x)
        return self.relu(x)


class ResGCNInputBranch(nn.Module):
    # 输入分支：处理 num_stream 个输入流（x/v/a）
    # 每个流 (N, C, T, V) 先映射到统一通道，再相加融合

    def __init__(self, num_stream, block, channels, A, reduction=4):
        super().__init__()
        self.num_stream = num_stream
        in_c, mid_c, out_c = channels
        self.branches = nn.ModuleList()
        for _ in range(num_stream):
            self.branches.append(nn.Sequential(
                nn.Conv2d(in_c, mid_c, 1),
                nn.BatchNorm2d(mid_c),
                nn.ReLU(inplace=True),
            ))
        # 融合后映射到 out_c
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_c, out_c, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: (N, num_stream, C, T, V)
        outs = [branch(x[:, i]) for i, branch in enumerate(self.branches)]
        fused = torch.stack(outs, dim=1).sum(dim=1)
        return self.fuse(fused)


class ResGCN(nn.Module):
    # ResGCN 骨干：输入分支 + 主分支 + 分类头

    def __init__(self, input_num=3, input_branch=None, main_stream=None,
                 num_class=74, reduction=4, block="Basic", graph=None):
        super().__init__()
        if graph is not None:
            self.register_buffer("A", graph.float())
        else:
            self.register_buffer("A", torch.ones(3, 17, 17))

        self.num_class = num_class
        self.block = block

        # 输入分支
        self.input_branch = ResGCNInputBranch(
            num_stream=input_num,
            block=input_branch["block"],
            channels=input_branch["channels"],
            A=self.A,
            reduction=reduction,
        )

        # 主分支
        main_cfgs = main_stream["channels"]
        in_c = input_branch["channels"][-1]
        self.main_stream = nn.ModuleList()
        for layer_idx, layer_cfg in enumerate(main_cfgs):
            layer_in = layer_cfg[0]
            blocks = []
            for j, out_c in enumerate(layer_cfg[1:]):
                stride = 2 if (j == 0 and layer_idx > 0) else 1
                blocks.append(ResGCNModule(
                    in_channels=layer_in, out_channels=out_c, block=block,
                    A=self.A, stride=stride, reduction=reduction,
                ))
                layer_in = out_c
            self.main_stream.append(nn.ModuleList(blocks))

        # 分类头：全局平均池化 -> FC
        final_c = main_cfgs[-1][-1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(final_c, num_class)

    def forward(self, x, return_feat=False):
        # x: (N, num_stream, C, T, V) 或 (N, C, T, V)
        if x.dim() == 4:
            x = x.unsqueeze(1)
        x = self.input_branch(x)          # (N, out_c, T, V)
        for layer in self.main_stream:
            for m in layer:
                x = m(x)
        pooled = self.global_pool(x).flatten(1)   # (N, final_c)
        logits = self.classifier(pooled)
        if return_feat:
            return logits, pooled
        return logits

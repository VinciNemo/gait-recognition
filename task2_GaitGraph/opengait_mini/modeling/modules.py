# 精简版 OpenGait —— 公共模块
# 对应 OpenGait 的 modules.py：Graph（时空图邻接矩阵）、空间/时间卷积块

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Graph:
    # 构建人体骨架时空图，参考 ST-GCN (AAAI 2018) 和 OpenGait 的 Graph
    # 输入 joint_format='coco'（17 个关键点），max_hop=3

    def __init__(self, joint_format="coco", max_hop=3, dilation=1):
        self.joint_format = joint_format
        self.max_hop = max_hop
        self.dilation = dilation
        self.num_node, self.edge, self.connect_joint, self.flip_idx, self.parts = self._get_edge()
        self.A = self._get_adjacency()

    def _get_edge(self):
        # COCO 17 关键点:
        # 0 nose, 1 l_eye, 2 r_eye, 3 l_ear, 4 r_ear,
        # 5 l_shoulder, 6 r_shoulder, 7 l_elbow, 8 r_elbow,
        # 9 l_wrist, 10 r_wrist, 11 l_hip, 12 r_hip,
        # 13 l_knee, 14 r_knee, 15 l_ankle, 16 r_ankle
        num_node = 17
        self_link = [(i, i) for i in range(num_node)]
        neighbor_link = [
            (0, 1), (0, 2), (1, 3), (2, 4),          # 头部
            (0, 5), (0, 6),                          # 鼻子-肩
            (5, 6),                                  # 左右肩
            (5, 7), (7, 9),                          # 左臂
            (6, 8), (8, 10),                         # 右臂
            (5, 11), (6, 12), (11, 12),              # 躯干
            (11, 13), (13, 15),                      # 左腿
            (12, 14), (14, 16),                      # 右腿
        ]
        edge = self_link + neighbor_link
        center = 5  # 中心点：左肩
        connect_joint = np.array([5, 0, 0, 1, 2, 0, 0, 5, 6, 7, 8, 5, 6, 11, 12, 13, 14])
        flip_idx = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
        parts = [
            np.array([5, 7, 9]),    # left_arm
            np.array([6, 8, 10]),   # right_arm
            np.array([11, 13, 15]), # left_leg
            np.array([12, 14, 16]), # right_leg
            np.array([0, 1, 2, 3, 4]),  # head
        ]
        return num_node, edge, connect_joint, flip_idx, parts

    def _get_hop_distance(self):
        hop_dis = np.full((self.num_node, self.num_node), np.inf)
        for u, v in self.edge:
            hop_dis[u, v] = 1
            hop_dis[v, u] = 1
        # Floyd 风格最短路径
        for k in range(self.num_node):
            for i in range(self.num_node):
                for j in range(self.num_node):
                    if hop_dis[i, j] > hop_dis[i, k] + hop_dis[k, j]:
                        hop_dis[i, j] = hop_dis[i, k] + hop_dis[k, j]
        return hop_dis

    def _normalize_digraph(self, A):
        Dl = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        return np.dot(A, Dn)

    def _get_adjacency(self):
        # 按 ST-GCN 的 spatial configuration 划分，邻接矩阵分成 3 个子集: root/close/further
        hop_dis = self._get_hop_distance()
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        A = np.zeros((len(valid_hop), self.num_node, self.num_node))
        for i, hop in enumerate(valid_hop):
            A[i][hop_dis == hop] = 1  # 距离为 hop 的节点连接
        # 归一化
        for i in range(len(valid_hop)):
            A[i] = self._normalize_digraph(A[i])
        return A


class SpatialGraphConv(nn.Module):
    # 空间图卷积：输入 (N, C, T, V)，邻接矩阵 A (K, V, V)

    def __init__(self, in_channels, out_channels, max_graph_distance):
        super().__init__()
        self.s_kernel_size = max_graph_distance + 1
        self.gcn = nn.Conv2d(in_channels, out_channels * self.s_kernel_size, 1)

    def forward(self, x, A):
        x = self.gcn(x)  # (N, K*C, T, V)
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v).contiguous()
        x = torch.einsum("nkctv,kvw->nctw", (x, A[:self.s_kernel_size])).contiguous()
        return x


class SpatialBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, max_graph_distance, reduction=4,
                 block_res=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.s_kernel_size = max_graph_distance + 1
        mid_channels = out_channels // reduction

        self.gcn = SpatialGraphConv(in_channels, out_channels, max_graph_distance)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if block_res:
            if in_channels != out_channels:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.residual = lambda x: x
        else:
            self.residual = lambda x: 0

    def forward(self, x, A):
        res = self.residual(x)
        x = self.gcn(x, A)
        x = self.bn(x)
        x = self.relu(x)
        x = x + res
        return x


class SpatialBottleneckBlock(nn.Module):
    # Bottleneck 空间图卷积（GaitGraph2/ResGCN 用）
    def __init__(self, in_channels, out_channels, max_graph_distance, reduction=4,
                 block_res=True):
        super().__init__()
        mid_channels = out_channels // reduction
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.gcn = SpatialGraphConv(mid_channels, mid_channels, max_graph_distance)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if block_res:
            if in_channels != out_channels:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.residual = lambda x: x
        else:
            self.residual = lambda x: 0

    def forward(self, x, A):
        res = self.residual(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.gcn(x, A)
        x = self.relu(self.bn2(x))
        x = self.bn3(self.conv3(x))
        return self.relu(x + res)


class TemporalBasicBlock(nn.Module):
    # 时间卷积块（TCN）
    def __init__(self, in_channels, out_channels, temporal_window_size=9, stride=1,
                 block_res=True, reduction=4):
        super().__init__()
        padding = (temporal_window_size - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=(temporal_window_size, 1),
                      stride=(stride, 1), padding=(padding, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.2, inplace=True),
        )
        if block_res:
            if in_channels != out_channels or stride != 1:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.residual = lambda x: x
        else:
            self.residual = lambda x: 0

    def forward(self, x):
        return self.tcn(x) + self.residual(x)


class TemporalBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_window_size=9, stride=1,
                 block_res=True, reduction=4):
        super().__init__()
        mid_channels = out_channels // reduction
        padding = (temporal_window_size - 1) // 2
        self.bn0 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels,
                               kernel_size=(temporal_window_size, 1),
                               stride=(stride, 1), padding=(padding, 0))
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.2, inplace=True)

        if block_res:
            if in_channels != out_channels or stride != 1:
                self.residual = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, (stride, 1)),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.residual = lambda x: x
        else:
            self.residual = lambda x: 0

    def forward(self, x):
        res = self.residual(x)
        x = self.relu(self.bn1(self.conv1(self.relu(self.bn0(x)))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(self.bn3(self.conv3(x)))
        return x + res

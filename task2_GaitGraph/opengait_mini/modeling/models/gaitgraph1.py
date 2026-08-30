# 任务2: 手动复现 GaitGraph1 —— 基于骨架的步态识别
# 对应 OpenGait 的 models/gaitgraph1.py，手工复现版
# 论文: GaitGraph (ICIP 2021)
# 思路:
#   1. 骨架序列看成时空图：空间上关节按人体结构相连（邻接矩阵 A），
#      时间上同一关节跨帧形成轨迹
#   2. 用 ResGCN（空间图卷积 + 时间卷积）提步态特征
#   3. 训练用监督对比学习 SupConLoss：序列切成前后两半当两个"视图"
#   4. 推理用 L2 归一化特征做度量匹配（Rank-1）
# 输入 ipts[0]: (N, T, V, I, C)，C = num_stream * 3（坐标/骨骼/运动各 3 通道）

import torch
import torch.nn.functional as F

from ..base_model import BaseModel
from ..backbones.resgcn import ResGCN
from ..modules import Graph


class GaitGraph1(BaseModel):
    # 手动复现版 GaitGraph1

    def __init__(self, model_cfg, **kwargs):
        self.graph = None
        self.ResGCN = None
        super().__init__(model_cfg, **kwargs)

    def build_network(self, model_cfg, **kwargs):
        self.joint_format = model_cfg["joint_format"]
        self.input_num = model_cfg["input_num"]
        self.block = model_cfg["block"]
        self.input_branch = model_cfg["input_branch"]
        self.main_stream = model_cfg["main_stream"]
        self.num_class = model_cfg["num_class"]
        self.reduction = model_cfg["reduction"]
        self.tta = model_cfg.get("tta", False)

        # 构建时空图（邻接矩阵 A）
        self.graph = Graph(joint_format=self.joint_format, max_hop=3)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)

        # ResGCN 骨干网络
        self.ResGCN = ResGCN(
            input_num=self.input_num,
            input_branch=self.input_branch,
            main_stream=self.main_stream,
            num_class=self.num_class,
            reduction=self.reduction,
            block=self.block,
            graph=A,
        )

    def _split_streams(self, x):
        # 把拼接输入 (N, T, V, I, C) 拆成 num_stream 个输入流，返回 (N, num_stream, 3, T, V)
        N, T, V, I, C = x.size()
        stream_c = C // self.input_num
        x = x.view(N, T, V, I, self.input_num, stream_c)  # (N,T,V,I,stream,3)
        x = x.permute(0, 4, 5, 1, 3, 2).contiguous()      # (N,stream,3,T,I,V)
        # 合并 I（人数）维度：取第一个人
        x = x[:, :, :, :, 0, :]                            # (N,stream,3,T,V)
        return x

    def forward(self, inputs):
        ipts, labs, type_, view_, seqL = inputs
        x_input = ipts[0]            # (N, T, V, I, C)
        N, T, V, I, C = x_input.size()
        pose = x_input

        if self.training:
            # 将一个序列在时间维切为两半，作为对比学习的两个"视图"
            # x: (N, stream, 3, T, V) -> (N, stream, 3, T/2, V) 两份
            x1 = self._split_streams(x_input[:, :T // 2])
            x2 = self._split_streams(x_input[:, T // 2:])
            f1 = self.ResGCN(x1, return_feat=True)[1]   # (N, feat)
            f2 = self.ResGCN(x2, return_feat=True)[1]
            f1 = F.normalize(f1, dim=1, p=2)
            f2 = F.normalize(f2, dim=1, p=2)
            embed = torch.stack([f1, f2], dim=1)        # (N, 2, feat)
            return {
                "training_feat": {
                    "SupConLoss": {"features": embed, "labels": labs},
                },
                "visual_summary": {
                    "image/pose": pose.view(N * T, 1, I * V, C).contiguous(),
                },
            }

        # 推理：整段序列 -> 特征
        x = self._split_streams(x_input)                # (N, stream, 3, T, V)
        _, feat = self.ResGCN(x, return_feat=True)      # (N, feat)
        feat = F.normalize(feat, dim=1, p=2)
        embed = feat.unsqueeze(-1)                      # (N, feat, 1)
        return {
            "inference_feat": {"embeddings": embed},
            "visual_summary": {
                "image/pose": pose.view(N * T, 1, I * V, C).contiguous(),
            },
        }

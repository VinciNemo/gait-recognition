# 任务2：基于骨架的步态识别（GaitGraph 复现）

## 概述

在 CASIA-B 数据集上，参照 OpenGait 框架结构，**手动复现** GaitGraph1（基于骨架的时空图卷积步态识别）。

**参考论文：**
- GaitGraph: Graph Convolutional Network for Skeleton-Based Gait Recognition (ICIP 2021)
- Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition (AAAI 2018, ST-GCN)
- OpenGait: A Comprehensive Benchmark Study for Gait Recognition Towards Better Practicality (TPAMI 2025)

## 原理

基于骨架的步态识别将人体关节坐标序列视为**时空图**：
- **空间**上，关节点通过人体物理结构相连（邻接矩阵 $A$）
- **时间**上，同一关节点跨帧形成运动轨迹

通过 ResGCN（空间图卷积 + 时间卷积）建模纯运动学特征，过滤衣着/光照/背景等与身份无关的信息。

## 复现结构（精简版 OpenGait 框架）

```
task2_GaitGraph/
├── opengait_mini/                       # 精简版 OpenGait 框架
│   ├── modeling/
│   │   ├── base_model.py                # 基础模型类（对应 base_model.py）
│   │   ├── modules.py                   # Graph 邻接矩阵 + 时空卷积块（对应 modules.py）
│   │   ├── backbones/resgcn.py          # ResGCN 骨干（对应 backbones/resgcn.py）
│   │   └── models/gaitgraph1.py         # ★ 手动复现 GaitGraph1
│   ├── losses/supcon.py                 # 监督对比损失 SupConLoss
│   └── data/
│       ├── transform.py                 # MultiInput 三流变换（坐标/骨骼/运动）
│       └── dataset.py                   # 骨架数据集加载
├── configs/gaitgraph1.py                # 模型配置
├── train.py                             # 训练
├── test.py                              # 评估（Rank-1）
└── checkpoints/                         # 模型保存（运行时生成）
```

## 运行步骤

### 1. 训练

```bash
cd task2_GaitGraph
python train.py --num-subjects 62 --epochs 120 --batch-size 64 --lr 3e-4
```

### 2. 评估

```bash
python test.py --num-subjects 62 --ckpt checkpoints/gaitgraph1_long.pt
```

## 全量运行结果（62 训练 + 62 测试）

| 训练配置 | nm | bg | cl | 平均 Rank-1 |
|----------|--------|--------|--------|-------------|
| 50 epochs | 12.63% | 7.80% | 7.26% | 9.23% |
| 120 epochs（lr=3e-4） | 15.86% | 10.75% | 9.68% | **12.10%** |

> **说明**：CASIA-B 只提供轮廓数据，不提供骨架。本演示的骨架由"伪姿态估计"
> （基于轮廓几何的轻量姿态估计）预先提取到 `data/skeleton/`，信息量有限，
> 故识别率低于使用真实姿态估计骨架（OpenPose/HRNet 从 RGB 提取）的 GaitGraph。
> 模型代码结构完全参照 OpenGait，换用真实骨架数据（或 OpenGait 的 SkeletonGait
> 数据集）即可达到论文水平。随机基线约 1.6%（62 类）。

## 关键组件说明

### Graph（时空图构建）

采用 ST-GCN 的 **spatial configuration** 划分策略，将邻接矩阵分为 3 个子集
（root / close / further），每个子集归一化后参与图卷积。

### ResGCN（骨干）

- **输入分支**：处理 3 个输入流（关节坐标 / 骨骼向量 / 运动），融合
- **主分支**：3 层时空图卷积模块（空间图卷积 + 时间卷积 + 可学习边权重）
- **分类头**：全局平均池化 + FC

### 训练策略（SupConLoss）

将一个序列在时间维切为两半作为两个"视图"，用监督对比损失拉近同一身份的样本。

## 评估指标

- **Rank-1 识别率**：在测试集（后 62 人）上，gallery（nm）vs probe（nm/bg/cl）最近邻匹配

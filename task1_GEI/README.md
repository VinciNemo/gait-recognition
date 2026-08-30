# 任务1：基于轮廓的经典步态识别（GEI 复现）

## 概述

复现 GEI（Gait Energy Image，步态能量图）特征构建方法，并在 CASIA-B 数据集上用 SVM / CNN 分类器做步态识别。

**参考论文：**
- GEINet: View-Invariant Gait Recognition Using a Convolutional Neural Network (ICB 2016 / IJCB)
- Individual Recognition Using Gait Energy Image (TPAMI 2006)

## 原理

GEI 将一个完整步态周期内、经过高度归一化和空间水平对齐的二值化人体轮廓图，按时间维度加权平均，压缩成一张单通道二维图像：

$$GEI(x, y) = \frac{1}{T} \sum_{t=1}^{T} S_t(x, y)$$

其中 $S_t$ 为第 $t$ 帧归一化后的二值轮廓，$T$ 为一个步态周期内的帧数。

## 文件结构

```
task1_GEI/
├── gei.py              # GEI 特征构建核心（归一化 + 时间平均）
├── dataset.py          # CASIA-B 数据加载、协议划分、GEI 缓存
├── svm_baseline.py     # GEI + RBF-SVM 分类识别
├── train_cnn.py        # GEI + CNN 分类识别（训练 + 评估）
├── visualize.py        # GEI 可视化
└── cache/              # GEI 缓存目录（运行时生成）
```

## 运行步骤

### 1. 生成 GEI 可视化（可选）

```bash
cd task1_GEI
python visualize.py --subject 1 --view 090
```

### 2. GEI + SVM

```bash
cd task1_GEI
python svm_baseline.py --num-subjects 62 --workers 4   # 全量（62 训练 + 62 测试）
```

### 3. GEI + CNN

```bash
cd task1_GEI
python train_cnn.py --num-subjects 62 --epochs 15 --workers 4
```

## 全量运行结果（62 训练 + 62 测试，前 62 / 后 62 人）

协议：训练用 `nm-01/02 × 000/090/180`；测试 gallery 用 `nm-01~04`，
probe 用 `nm / bg / cl`（各 372 个探针序列）。

| 方法 | nm | bg | cl | 平均 Rank-1 |
|------|--------|--------|--------|-------------|
| **GEI + SVM**（PCA→128维 + RBF） | 98.92% | 57.80% | 12.63% | **56.45%** |
| **GEI + CNN**（GPU, 15 epochs） | 90.59% | 40.05% | 7.80% | **46.15%** |

符合 CASIA-B 经典规律：nm > bg > cl。

## 评估指标

- **Rank-1 识别率**：gallery-probe 最近邻匹配（开放集度量识别）
  - 本协议训练/测试身份不相交（前 62 vs 后 62 人），因此不使用封闭集分类

## 说明

- 默认使用 CASIA-B 经典协议：前 62 人训练，后 62 人测试
- 训练序列 nm-01/02，测试 gallery 用 nm-01~04，probe 用 nm/bg/cl
- `--num-subjects` 控制使用人数（62 = 全量）
- 性能注意：Windows 下 sklearn 的 RBF-SVM / scipy PCA 在 4096 维会原生崩溃，
  本实现用 numpy SVD 自定义 PCA 降到 128 维后训练 RBF-SVM

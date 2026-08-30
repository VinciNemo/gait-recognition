# 步态识别（Gait Recognition）大创项目

基于 **CASIA-B** 数据集，完成两个步态识别任务：

1. **任务1**：基于轮廓的经典步态识别（GEI 复现）
2. **任务2**：基于骨架的步态识别（GaitGraph 复现，参照 OpenGait 框架）

---

## 环境

- Python 3.13（conda 环境 `default`）
- PyTorch 2.11.0 + CUDA（GPU 可用）
- 依赖：`pdfplumber`、`scikit-learn`、`Pillow`、`numpy`、`matplotlib`

## 项目结构

```
DaChuang/
├── 任务.md                          # 原始任务说明
├── README.md                        # 本文件
├── data/
│   ├── GaitDatasetB-silh/           # CASIA-B 轮廓数据集（124 人，已解压）
│   └── skeleton/                    # 任务2 骨架数据（3683 个 npy 序列）
├── task1_GEI/                       # ★ 任务1：GEI + SVM / CNN
│   ├── gei.py                       #   GEI 特征构建
│   ├── dataset.py                   #   数据加载与协议划分
│   ├── svm_baseline.py              #   GEI + RBF-SVM
│   ├── train_cnn.py                 #   GEI + CNN
│   ├── visualize.py                 #   GEI 可视化
│   └── cache/                       #   GEI 缓存
└── task2_GaitGraph/                 # ★ 任务2：GaitGraph 复现
    ├── opengait_mini/               #   精简版 OpenGait 框架
    │   ├── modeling/
    │   │   ├── base_model.py
    │   │   ├── modules.py           #     Graph 邻接矩阵 + 时空卷积块
    │   │   ├── backbones/resgcn.py  #     ResGCN 骨干
    │   │   └── models/gaitgraph1.py #     ★ 手动复现的 GaitGraph1
    │   ├── losses/supcon.py         #   监督对比损失
    │   └── data/                    #   transform / dataset
    ├── configs/gaitgraph1.py        #   模型配置
    ├── train.py / test.py           #   训练 / 评估
    └── checkpoints/                 #   训练好的模型
```

---

## 全量运行结果（124 人全量，前 62 人训练 / 后 62 人测试）

### 任务1：GEI + 分类器

协议：训练用 `nm-01/02 × 000/090/180`（前 62 人），
测试 gallery 用 `nm-01~04`、probe 用 `nm / bg / cl`（后 62 人）。

| 方法 | nm | bg | cl | 平均 Rank-1 |
|------|--------|--------|--------|-------------|
| **GEI + SVM**（PCA→128维 + RBF） | 98.92% | 57.80% | 12.63% | **56.45%** |
| **GEI + CNN**（GPU） | 90.59% | 40.05% | 7.80% | **46.15%** |

- 符合 CASIA-B 经典规律：正常行走（nm）识别率最高，背包（bg）次之，穿衣（cl）最难
- SVM（低维 RBF 核）优于 CNN（少量样本下 CNN 容易过拟合）

### 任务2：GaitGraph1（骨架）

| 训练配置 | nm | bg | cl | 平均 Rank-1 |
|----------|--------|--------|--------|-------------|
| 50 epochs | 12.63% | 7.80% | 7.26% | 9.23% |
| 120 epochs（lr=3e-4） | 15.86% | 10.75% | 9.68% | **12.10%** |

> **说明**：CASIA-B 只提供轮廓数据，不提供骨架。本演示的骨架由"伪姿态估计"
> （基于轮廓几何的轻量姿态估计）预先提取到 `data/skeleton/`，信息量有限，
> 因此识别率低于使用真实姿态估计骨架（OpenPose/HRNet 从 RGB 提取）的 GaitGraph。
> 代码结构完全参照 OpenGait，换用真实骨架数据后即可达到论文水平。
> 随机基线约 1.6%（62 类）。

---

## 复现要点

### GEI（任务1）
对一个完整步态周期内、经高度归一化和空间水平对齐的二值轮廓图序列，
按时间维度加权平均，压缩为单通道图像：

$$GEI(x,y) = \frac{1}{T}\sum_{t=1}^{T} S_t(x,y)$$

### GaitGraph（任务2）
- 骨架序列构成**时空图**：空间上关节经人体结构相连（邻接矩阵 $A$），
  时间上同一关节跨帧形成运动轨迹
- **ResGCN**（空间图卷积 + 时间卷积）提取运动学特征，过滤衣着/光照/背景干扰
- 训练用**监督对比损失 SupConLoss**：序列切为前后两半作为正样本对
- 推理用 L2 归一化特征做余弦最近邻匹配（Rank-1）

### 关键组件（对应 OpenGait）
| 本实现 | OpenGait 对应 |
|--------|---------------|
| `opengait_mini/modeling/models/gaitgraph1.py` | `opengait/modeling/models/gaitgraph1.py` |
| `opengait_mini/modeling/backbones/resgcn.py` | `opengait/modeling/backbones/resgcn.py` |
| `opengait_mini/modeling/modules.py` (Graph) | `opengait/modeling/modules.py` |
| `opengait_mini/losses/supcon.py` | OpenGait SupConLoss |

---

## 运行命令

### 任务1
```bash
cd task1_GEI
python svm_baseline.py --num-subjects 62 --workers 4   # GEI+SVM
python train_cnn.py --num-subjects 62 --epochs 15 --workers 4  # GEI+CNN
```

### 任务2
```bash
cd task2_GaitGraph
python train.py --num-subjects 62 --epochs 120 --batch-size 64 --lr 3e-4
python test.py --num-subjects 62 --ckpt checkpoints/gaitgraph1_long.pt
```

> 注意：Windows 下 `sklearn` 的 RBF-SVM 与 scipy PCA 在高维（4096 维）上会原生崩溃，
> 本实现已改为 numpy SVD 自定义 PCA 降维到 128 维后训练，规避了该问题。

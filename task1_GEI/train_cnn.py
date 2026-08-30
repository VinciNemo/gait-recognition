# 任务1: GEI + CNN 分类识别
# 参考 GEINet (ICB 2016) 的思路，用了个更轻量的 CNN：
# 输入 GEI (1, H, W) -> 几层卷积 -> 全局池化 -> 全连接分类
# 交叉熵训练，在测试集上评估

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import (find_data_root, build_gei_dataset,
                     TRAIN_SUBJECTS, TEST_SUBJECTS,
                     TRAIN_SEQ_TYPES, TRAIN_VIEWS,
                     GALLERY_SEQ_TYPES, GALLERY_VIEWS,
                     PROBE_SEQ_TYPES, PROBE_VIEWS)

DEFAULT_DATA_ROOTS = [Path("../data"), Path("../data/GaitDatasetB-silh")]


class GeiCNN(nn.Module):
    # 轻量 CNN，输入 GEI (B, 1, H, W)

    def __init__(self, num_classes, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # -> 32x32

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # -> 16x16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # -> 8x8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, return_feat=False):
        feat = self.features(x)
        pooled = self.global_pool(feat).flatten(1)
        out = self.classifier(pooled)
        if return_feat:
            return out, pooled
        return out


def parse_args():
    parser = argparse.ArgumentParser(description="GEI + CNN 分类识别")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--num-subjects", type=int, default=30)
    parser.add_argument("--target-size", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--cache", type=str, default="cache")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4,
                        help="GEI 构建并行进程数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    if args.device != "auto":
        device = args.device
    print(f"设备: {device}")

    roots = [Path(args.data_root)] if args.data_root else DEFAULT_DATA_ROOTS
    root = find_data_root(roots)
    print(f"数据集根目录: {root}")

    cache_dir = Path(args.cache)
    cache_dir.mkdir(exist_ok=True)
    size = (args.target_size, args.target_size)

    train_subs = list(rng.choice(TRAIN_SUBJECTS, args.num_subjects, replace=False))
    test_subs = list(rng.choice(TEST_SUBJECTS, args.num_subjects, replace=False))
    print(f"训练 subject: {len(train_subs)} 个, 测试 subject: {len(test_subs)} 个")

    # ---------- 数据 ----------
    print("\n[1/4] 构建 GEI 数据集 ...")
    train = build_gei_dataset(root, train_subs, TRAIN_SEQ_TYPES, TRAIN_VIEWS,
                              size, args.max_frames,
                              cache_dir / f"train_s{args.num_subjects}.pkl",
                              workers=args.workers)
    X_train, y_train = train["X"], train["y"]
    print(f"  训练 GEI: {len(X_train)} 张")

    gallery = build_gei_dataset(root, test_subs, GALLERY_SEQ_TYPES, GALLERY_VIEWS,
                                size, args.max_frames,
                                cache_dir / f"gallery_s{args.num_subjects}.pkl",
                                workers=args.workers)
    probes = {}
    for name, seq_types in PROBE_SEQ_TYPES.items():
        probes[name] = build_gei_dataset(root, test_subs, seq_types, PROBE_VIEWS,
                                         size, args.max_frames,
                                         cache_dir / f"probe_{name}_s{args.num_subjects}.pkl",
                                         workers=args.workers)

    # ---------- 训练 ----------
    print("\n[2/4] 训练 CNN ...")
    X_t = torch.from_numpy(X_train[:, None].astype(np.float32))
    y_t = torch.from_numpy(y_train.astype(np.int64))
    # 将 subject id 映射到 0..K-1
    classes = np.unique(y_train)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = torch.tensor([class_to_idx[int(c)] for c in y_train], dtype=torch.long)
    dataset = TensorDataset(X_t, y_idx)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=False)

    model = GeiCNN(num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(xb)
        scheduler.step()
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={total_loss/total:.4f}  "
                  f"train_acc={correct/total*100:.2f}%")
    print(f"  CNN 训练完成, 用时 {time.time() - t0:.1f}s")

    # ---------- 特征提取 + 评估 ----------
    print("\n[3/4] 提取测试特征 ...")
    model.eval()

    @torch.no_grad()
    def extract_features(X):
        X = torch.from_numpy(X[:, None].astype(np.float32))
        feats = []
        for i in range(0, len(X), 128):
            batch = X[i:i + 128].to(device)
            _, pooled = model(batch, return_feat=True)
            feats.append(pooled.cpu().numpy())
        return np.concatenate(feats, axis=0)

    gal_feat = extract_features(gallery["X"])
    probe_feat = {name: extract_features(probes[name]["X"]) for name in probes}

    def rank1(probe_f, probe_y):
        # 余弦相似度最近邻 -> subject id
        gal_y = gallery["y"]
        sims = probe_f @ gal_feat.T
        norm_p = np.linalg.norm(probe_f, axis=1, keepdims=True)   # (P,1)
        norm_g = np.linalg.norm(gal_feat, axis=1)[None, :]        # (1,G)
        sims = sims / (norm_p * norm_g + 1e-9)
        pred = gal_y[sims.argmax(axis=1)]
        return (pred == probe_y).mean()

    print("\n[4/4] 评估 (gallery-probe 识别, Rank-1):")
    print("  注: 本协议训练/测试身份不相交（前 62 vs 后 62 人），"
          "属于开放集度量识别，使用 gallery-probe 最近邻匹配。")
    mean = 0.0
    for name in ["nm", "bg", "cl"]:
        acc = rank1(probe_feat[name], probes[name]["y"])
        mean += acc / 3
        print(f"  [{name}] Rank-1: {acc * 100:.2f}%  (probe={len(probe_feat[name])})")

    print(f"\n[完成] CNN 平均 Rank-1: {mean * 100:.2f}%")


if __name__ == "__main__":
    main()

# 任务2: 训练 GaitGraph1（基于骨架的步态识别）
# 在提取好的骨架数据上用 SupConLoss 训练
# 用法: python train.py --skeleton-dir ../data/skeleton --num-subjects 20 --epochs 20

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# 把 opengait_mini 加进搜索路径
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from opengait_mini.modeling.models.gaitgraph1 import GaitGraph1
from opengait_mini.losses.supcon import SupConLoss
from opengait_mini.data.transform import MultiInput
from opengait_mini.data.dataset import (SkeletonDataset, build_dataloader,
                                        TRAIN_SUBJECTS, TRAIN_SEQ_TYPES, TRAIN_VIEWS)
from configs.gaitgraph1 import MODEL_CFG, TRAIN_CFG, DATA_CFG


def parse_args():
    parser = argparse.ArgumentParser(description="训练 GaitGraph1")
    parser.add_argument("--skeleton-dir", type=str, default="../data/skeleton",
                        help="骨架数据目录（含 .npy 文件）")
    parser.add_argument("--num-subjects", type=int, default=20,
                        help="训练使用的 subject 数量（小规模快速演示）")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seq-frames", type=int, default=40)
    parser.add_argument("--ckpt", type=str, default="checkpoints/gaitgraph1.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    if args.device != "auto":
        device = args.device
    print(f"设备: {device}")

    skeleton_dir = Path(args.skeleton_dir)
    if not skeleton_dir.exists():
        print(f"[错误] 骨架数据目录不存在: {skeleton_dir}")
        print("请先运行 extract_skeleton.py 从 CASIA-B 轮廓提取骨架。")
        sys.exit(1)

    # 训练 subject 划分
    train_subs = list(rng.choice(TRAIN_SUBJECTS, args.num_subjects, replace=False))
    print(f"训练 subject: {len(train_subs)} 个: {train_subs[:5]}...")

    # 数据集
    transform = MultiInput(center=DATA_CFG["center"])
    dataset = SkeletonDataset(skeleton_dir, train_subs, TRAIN_SEQ_TYPES, TRAIN_VIEWS,
                              seq_frames=args.seq_frames, transform=transform)
    print(f"训练序列数: {len(dataset)}")

    # 模型（num_class 设为训练身份数，实际训练只用 SupConLoss）
    model_cfg = dict(MODEL_CFG)
    model_cfg["num_class"] = len(train_subs)
    model = GaitGraph1(model_cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = SupConLoss(temperature=TRAIN_CFG["temperature"])

    # 训练
    print("\n开始训练 ...")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total = 0.0, 0
        for xb, yb in build_dataloader(dataset, args.batch_size, shuffle=True, seed=epoch):
            xb = torch.from_numpy(xb).float().to(device)  # (N, T, V, 1, C)
            yb = torch.from_numpy(yb).long().to(device)
            retval = model(([xb], yb, None, None, None))
            loss = criterion(**retval["training_feat"]["SupConLoss"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
            total += len(xb)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={total_loss/max(total,1):.4f}")

    print(f"训练完成, 用时 {time.time()-t0:.1f}s")

    # 保存
    ckpt_dir = Path(args.ckpt).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(),
                "model_cfg": model_cfg,
                "train_subs": train_subs}, args.ckpt)
    print(f"模型已保存: {args.ckpt}")


if __name__ == "__main__":
    main()

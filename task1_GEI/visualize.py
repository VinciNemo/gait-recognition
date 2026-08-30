# 可视化 GEI：生成某个人某个视角下的 GEI，跟原始轮廓帧对比
# 用法: python visualize.py --subject 1 --view 090

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import find_data_root, find_sequence_dirs
from gei import build_gei_from_sequence, load_silhouette_frame

DEFAULT_DATA_ROOTS = [Path("../data"), Path("../data/GaitDatasetB-silh")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--seq", type=str, default="nm-01")
    parser.add_argument("--view", type=str, default="090")
    parser.add_argument("--out", type=str, default="gei_demo.png")
    args = parser.parse_args()

    roots = [Path(args.data_root)] if args.data_root else DEFAULT_DATA_ROOTS
    root = find_data_root(roots)
    subj_dir = root / f"{args.subject:03d}"
    seqs = find_sequence_dirs(subj_dir)
    seq_dir = seqs.get((args.seq, args.view))
    if seq_dir is None:
        print(f"未找到序列 {args.subject:03d}/{args.seq}/{args.view}")
        sys.exit(1)

    pngs = sorted(seq_dir.glob("*.png"))
    print(f"序列帧数: {len(pngs)}, 序列目录: {seq_dir}")

    gei = build_gei_from_sequence(seq_dir, target_size=(64, 64), max_frames=80)

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes[0]):
        idx = int(i * (len(pngs) - 1) / 3)
        frame = load_silhouette_frame(pngs[idx], (64, 64))
        ax.imshow(frame, cmap="gray")
        ax.set_title(f"frame {idx}")
        ax.axis("off")

    axes[1, 0].imshow(gei, cmap="gray")
    axes[1, 0].set_title("GEI (64x64)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(gei, cmap="viridis")
    axes[1, 1].set_title("GEI (colormap)")
    axes[1, 1].axis("off")

    # GEI 统计
    axes[1, 2].axis("off")
    axes[1, 2].text(0.1, 0.5, f"shape: {gei.shape}\n"
                                f"mean: {gei.mean():.3f}\n"
                                f"max: {gei.max():.3f}\n"
                                f"前景占比: {(gei > 0.1).mean()*100:.1f}%",
                    fontsize=11, va="center")

    axes[1, 3].axis("off")
    axes[1, 3].text(0.1, 0.5, f"subject={args.subject:03d}\nseq={args.seq}\n"
                              f"view={args.view}°", fontsize=12, va="center")

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"已保存: {args.out}")


if __name__ == "__main__":
    main()

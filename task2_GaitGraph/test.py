# 任务2: 评估 GaitGraph1（Rank-1 识别率）
# 在测试集（后 62 人）上用训练好的模型提骨架特征，做 gallery-probe 匹配，
# 输出 nm / bg / cl 三种条件下的 Rank-1
# 用法: python test.py --skeleton-dir ../data/skeleton --ckpt checkpoints/gaitgraph1.pt

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from opengait_mini.modeling.models.gaitgraph1 import GaitGraph1
from opengait_mini.data.transform import MultiInput
from opengait_mini.data.dataset import (SkeletonDataset, load_sequence, collate_sequences,
                                        TEST_SUBJECTS, GALLERY_SEQ_TYPES, GALLERY_VIEWS,
                                        PROBE_SEQ_TYPES, PROBE_VIEWS)
from configs.gaitgraph1 import MODEL_CFG, DATA_CFG


def parse_args():
    parser = argparse.ArgumentParser(description="评估 GaitGraph1")
    parser.add_argument("--skeleton-dir", type=str, default="../data/skeleton")
    parser.add_argument("--ckpt", type=str, default="checkpoints/gaitgraph1.pt")
    parser.add_argument("--num-subjects", type=int, default=20,
                        help="测试使用的 subject 数量（小规模快速演示）")
    parser.add_argument("--seq-frames", type=int, default=40)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-cuda", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def extract_embeddings(model, items, transform, seq_frames, device, batch=32):
    # 提取序列特征，返回 (N, feat)
    feats = []
    for i in range(0, len(items), batch):
        seqs = []
        for path, *_ in items[i:i + batch]:
            sk = load_sequence(path, seq_frames)
            if transform is not None:
                sk = transform(sk)
            seqs.append(sk)
        xb = collate_sequences(seqs)
        xb = torch.from_numpy(xb).float().to(device)
        retval = model(([xb], None, None, None, None))
        emb = retval["inference_feat"]["embeddings"]  # (N, feat, 1)
        feats.append(emb.squeeze(-1).cpu().numpy())
    return np.concatenate(feats, axis=0)


def rank1_eval(probe_feat, probe_labels, gallery_feat, gallery_labels):
    # 余弦相似度最近邻匹配，返回 Rank-1
    probe_feat = probe_feat / (np.linalg.norm(probe_feat, axis=1, keepdims=True) + 1e-9)
    gallery_feat = gallery_feat / (np.linalg.norm(gallery_feat, axis=1, keepdims=True) + 1e-9)
    sims = probe_feat @ gallery_feat.T
    pred = gallery_labels[sims.argmax(axis=1)]
    return (pred == probe_labels).mean()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    if args.device != "auto":
        device = args.device
    print(f"设备: {device}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model_cfg = ckpt["model_cfg"]
    model = GaitGraph1(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"已加载模型: {args.ckpt}")

    skeleton_dir = Path(args.skeleton_dir)
    rng = np.random.default_rng(0)
    test_subs = list(rng.choice(TEST_SUBJECTS, args.num_subjects, replace=False))
    print(f"测试 subject: {len(test_subs)} 个")

    transform = MultiInput(center=DATA_CFG["center"])

    # gallery: 用 nm-01~04 序列
    gal_ds = SkeletonDataset(skeleton_dir, test_subs, GALLERY_SEQ_TYPES, GALLERY_VIEWS,
                             seq_frames=args.seq_frames, transform=None)
    gal_feat = extract_embeddings(model, gal_ds.items, transform, args.seq_frames, device)
    gal_labels = np.array([subj for _, subj, _, _ in gal_ds.items])
    print(f"gallery: {len(gal_feat)} 序列")

    # probe: nm / bg / cl
    print("\nRank-1 识别率:")
    mean = 0.0
    for name, seq_types in PROBE_SEQ_TYPES.items():
        pr_ds = SkeletonDataset(skeleton_dir, test_subs, seq_types, PROBE_VIEWS,
                                seq_frames=args.seq_frames, transform=None)
        pr_feat = extract_embeddings(model, pr_ds.items, transform, args.seq_frames, device)
        pr_labels = np.array([subj for _, subj, _, _ in pr_ds.items])
        acc = rank1_eval(pr_feat, pr_labels, gal_feat, gal_labels)
        mean += acc / len(PROBE_SEQ_TYPES)
        print(f"  [{name}] Rank-1: {acc*100:.2f}%  (probe={len(pr_feat)})")

    print(f"\n[完成] 平均 Rank-1: {mean*100:.2f}%")


if __name__ == "__main__":
    main()

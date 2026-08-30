# 任务1: GEI + SVM 分类识别
# 用 GEI 展平向量当特征训练 RBF-SVM，在 CASIA-B 测试集（后 62 人）上评估
# 加 --num-subjects 可以只跑部分人，快速走通全流程

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

from dataset import (find_data_root, build_gei_dataset,
                     TRAIN_SUBJECTS, TEST_SUBJECTS,
                     TRAIN_SEQ_TYPES, TRAIN_VIEWS,
                     GALLERY_SEQ_TYPES, GALLERY_VIEWS,
                     PROBE_SEQ_TYPES, PROBE_VIEWS)

DEFAULT_DATA_ROOTS = [
    Path("../data"),
    Path("../data/GaitDatasetB-silh"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="GEI + SVM 分类识别")
    parser.add_argument("--data-root", type=str, default=None,
                        help="CASIA-B 数据集根目录，默认自动搜索 ../data")
    parser.add_argument("--num-subjects", type=int, default=30,
                        help="每个 split 使用的 subject 数量（小规模快速演示），"
                             "填 0 表示用全部")
    parser.add_argument("--target-size", type=int, default=64, help="GEI 尺寸 (H=W)")
    parser.add_argument("--max-frames", type=int, default=80,
                        help="每个序列最多使用的帧数")
    parser.add_argument("--cache", type=str, default="cache",
                        help="GEI 缓存目录")
    parser.add_argument("--pca-dim", type=int, default=128,
                        help="PCA 降维维数")
    parser.add_argument("--workers", type=int, default=4,
                        help="GEI 构建并行进程数")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def pca_fit(X, n_components=128):
    # 自己写的 PCA（基于 numpy.linalg.svd）
    # sklearn 的 PCA 内部调 scipy.linalg.svd，在 Windows/OpenBLAS 下对高维输入会崩，
    # numpy 的 svd 没事，所以干脆自己实现
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # numpy SVD 对本数据规模稳定
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]   # (n_components, n_features)
    return mean, components


def pca_transform(X, mean, components):
    return (X - mean) @ components.T


def evaluate_gallery_probe(features, labels):
    # gallery-probe 识别：每个 probe 去 gallery 里找最近的身份，算 Rank-1
    from sklearn.metrics import pairwise_distances
    gallery_X = features["gallery"]
    gallery_y = np.array(labels["gallery"])
    results = {}
    for name in ["nm", "bg", "cl"]:
        probe_X = features[f"probe_{name}"]
        probe_y = np.array(labels[f"probe_{name}"])
        dists = pairwise_distances(probe_X, gallery_X, metric="euclidean")
        # 最近邻的 subject id
        pred = gallery_y[dists.argmin(axis=1)]
        acc = (pred == probe_y).mean()
        results[name] = acc
        print(f"  [{name}] Rank-1 识别率: {acc * 100:.2f}%  (probe={len(probe_y)})")
    mean = float(np.mean(list(results.values())))
    results["mean"] = mean
    return results


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    roots = [Path(args.data_root)] if args.data_root else DEFAULT_DATA_ROOTS
    root = find_data_root(roots)
    print(f"数据集根目录: {root}")

    cache_dir = Path(args.cache)
    cache_dir.mkdir(exist_ok=True)
    size = (args.target_size, args.target_size)

    # 训练/测试 subject 划分
    train_subs = TRAIN_SUBJECTS
    test_subs = TEST_SUBJECTS
    if args.num_subjects > 0:
        train_subs = list(rng.choice(TRAIN_SUBJECTS, args.num_subjects, replace=False))
        test_subs = list(rng.choice(TEST_SUBJECTS, args.num_subjects, replace=False))
    print(f"训练 subject: {len(train_subs)} 个, 测试 subject: {len(test_subs)} 个")

    # ---------- 构建训练集 ----------
    print("\n[1/4] 构建训练集 GEI ...")
    t0 = time.time()
    train = build_gei_dataset(root, train_subs, TRAIN_SEQ_TYPES, TRAIN_VIEWS,
                              size, args.max_frames,
                              cache_dir / f"train_s{args.num_subjects}.pkl",
                              workers=args.workers)
    X_train, y_train = train["X"], train["y"]
    print(f"  训练 GEI 数量: {len(X_train)}, 用时 {time.time() - t0:.1f}s, 类别数: {len(np.unique(y_train))}")

    # ---------- 训练 SVM ----------
    print("\n[2/4] 训练 RBF-SVM（PCA 降维后） ...")
    X_flat = X_train.reshape(len(X_train), -1).astype(np.float32)
    scaler = StandardScaler().fit(X_flat)
    X_scaled = scaler.transform(X_flat)

    # PCA 降维：4096 维 RBF 核在 Windows/OpenBLAS 下会崩溃，
    # 且高维 RBF 核易过拟合、速度慢。降维到 128 维（标准做法）。
    pca_mean, pca_components = pca_fit(X_scaled, n_components=args.pca_dim)
    X_pca = pca_transform(X_scaled, pca_mean, pca_components)
    print(f"  PCA 降维: {X_scaled.shape[1]} -> {X_pca.shape[1]} 维")

    t0 = time.time()
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", random_state=args.seed)
    svm.fit(X_pca, y_train)
    print(f"  SVM 训练完成, 用时 {time.time() - t0:.1f}s, 支持向量数: {len(svm.support_)}")

    # ---------- 构建测试集（gallery + probe） ----------
    print("\n[3/4] 构建测试集 GEI ...")
    features = {}
    labels = {}

    gallery = build_gei_dataset(root, test_subs, GALLERY_SEQ_TYPES, GALLERY_VIEWS,
                                size, args.max_frames,
                                cache_dir / f"gallery_s{args.num_subjects}.pkl",
                                workers=args.workers)
    features["gallery"] = gallery["X"].reshape(len(gallery["X"]), -1)
    labels["gallery"] = gallery["y"]

    for name, seq_types in PROBE_SEQ_TYPES.items():
        probe = build_gei_dataset(root, test_subs, seq_types, PROBE_VIEWS,
                                  size, args.max_frames,
                                  cache_dir / f"probe_{name}_s{args.num_subjects}.pkl",
                                  workers=args.workers)
        features[f"probe_{name}"] = probe["X"].reshape(len(probe["X"]), -1)
        labels[f"probe_{name}"] = probe["y"]
    print(f"  gallery={len(features['gallery'])}, probe nm/bg/cl = "
          f"{len(features['probe_nm'])}/{len(features['probe_bg'])}/{len(features['probe_cl'])}")

    # ---------- 评估 ----------
    print("\n[4/4] 评估 (gallery-probe 识别, Rank-1):")
    print("  注: 本协议训练/测试身份不相交（前 62 vs 后 62 人），"
          "属于开放集度量识别，使用 gallery-probe 最近邻匹配。")
    results = evaluate_gallery_probe(features, labels)

    print(f"\n[完成] SVM 平均 Rank-1: {results['mean'] * 100:.2f}%")


if __name__ == "__main__":
    main()

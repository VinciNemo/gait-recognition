# CASIA-B 数据集加载与 GEI 数据集生成
# 目录结构: 人/序列类型(nm/bg/cl)/视角(000~180)/帧图
#   GaitDatasetB-silh/001/nm-01/000/000.png ...

import numpy as np
import pickle
from pathlib import Path

from gei import build_gei_from_sequence

# CASIA-B 数据集划分协议（经典协议，如 GaitSet / GEINet）
TRAIN_SUBJECTS = list(range(1, 63))       # 前 62 人作为训练集
TEST_SUBJECTS = list(range(63, 125))      # 后 62 人作为测试集

# 训练用的序列类型与视角
TRAIN_SEQ_TYPES = ["nm-01", "nm-02"]
TRAIN_VIEWS = ["000", "090", "180"]

# 测试：gallery 用 nm 序列，probe 用 nm/bg/cl
GALLERY_SEQ_TYPES = ["nm-01", "nm-02", "nm-03", "nm-04"]
GALLERY_VIEWS = ["000", "090", "180"]
PROBE_SEQ_TYPES = {
    "nm": ["nm-05", "nm-06"],
    "bg": ["bg-01", "bg-02"],
    "cl": ["cl-01", "cl-02"],
}
PROBE_VIEWS = ["000", "090", "180"]


def find_data_root(roots):
    # 在候选路径里找数据集根目录（解压后顶层目录名可能不一样）
    for root in roots:
        if not root.exists():
            continue
        # 检查 root 下是否有形如 001/ 的 subject 目录
        subjects = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
        if subjects:
            return root
    raise FileNotFoundError(f"未找到 CASIA-B 数据集，请在以下位置之一放置: {[str(r) for r in roots]}")


def scan_subjects(root):
    # 扫描所有 subject 目录，返回 {subject_id: 目录}
    return {p.name: p for p in sorted(root.iterdir())
            if p.is_dir() and p.name.isdigit()}


def find_sequence_dirs(subject_dir):
    # 返回 {(序列类型, 视角): 序列目录}
    result = {}
    for seq_dir in subject_dir.iterdir():
        if not seq_dir.is_dir():
            continue
        seq_type = seq_dir.name
        for view_dir in seq_dir.iterdir():
            if view_dir.is_dir():
                result[(seq_type, view_dir.name)] = view_dir
    return result


def _gei_worker(task):
    # 多进程 worker，构建单个 GEI
    from gei import build_gei_from_sequence
    seq_dir, target_size, max_frames, sid, seq_type, view = task
    try:
        gei = build_gei_from_sequence(Path(seq_dir), tuple(target_size), max_frames)
        return gei, sid, seq_type, view, None
    except Exception as e:
        return None, sid, seq_type, view, str(e)


def build_gei_dataset(root, subject_ids,
                      seq_types, views,
                      target_size=(64, 64),
                      max_frames=80,
                      cache_path=None,
                      workers=4):
    # 给指定的 subject/序列类型/视角构建 GEI 数据集，支持多进程
    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # 收集任务
    tasks = []
    for sid in subject_ids:
        subj_dir = root / f"{sid:03d}"
        if not subj_dir.exists():
            print(f"[warn] 缺少 subject {sid}")
            continue
        seqs = find_sequence_dirs(subj_dir)
        for seq_type in seq_types:
            for view in views:
                seq_dir = seqs.get((seq_type, view))
                if seq_dir is not None:
                    tasks.append((seq_dir, target_size, max_frames, int(sid), seq_type, view))

    X, y, meta = [], [], []
    if workers > 1 and len(tasks) > 1:
        import concurrent.futures
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            for i, (gei, sid, seq_type, view, err) in enumerate(
                    ex.map(_gei_worker, tasks), 1):
                if gei is None:
                    print(f"[warn] 构建 GEI 失败 {sid}/{seq_type}/{view}: {err}")
                    continue
                X.append(gei)
                y.append(sid)
                meta.append((sid, seq_type, view))
                if i % 300 == 0:
                    print(f"  GEI 构建进度: {i}/{len(tasks)}")
    else:
        for task in tasks:
            gei, sid, seq_type, view, err = _gei_worker(task)
            if gei is None:
                print(f"[warn] 构建 GEI 失败 {sid}/{seq_type}/{view}: {err}")
                continue
            X.append(gei)
            y.append(sid)
            meta.append((sid, seq_type, view))

    data = {"X": np.stack(X), "y": np.array(y), "meta": meta}
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
    return data


def build_gallery_probe(root, target_size=(64, 64), max_frames=80, cache_dir=None):
    # 构建测试用的 gallery 和 nm/bg/cl 三种 probe
    result = {}
    gallery_cache = cache_dir / "gallery.pkl" if cache_dir else None
    result["gallery"] = build_gei_dataset(
        root, TEST_SUBJECTS, GALLERY_SEQ_TYPES, GALLERY_VIEWS,
        target_size, max_frames, gallery_cache)

    result["probe"] = {}
    for name, seq_types in PROBE_SEQ_TYPES.items():
        probe_cache = cache_dir / f"probe_{name}.pkl" if cache_dir else None
        result["probe"][name] = build_gei_dataset(
            root, TEST_SUBJECTS, seq_types, PROBE_VIEWS,
            target_size, max_frames, probe_cache)
    return result


if __name__ == "__main__":
    import sys
    data_root = find_data_root([Path("../data"), Path("../data/GaitDatasetB-silh")])
    print(f"数据集根目录: {data_root}")
    subjects = scan_subjects(data_root)
    print(f"subject 数量: {len(subjects)}")

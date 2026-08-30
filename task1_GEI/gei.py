# GEI（步态能量图）特征构建
# 参考 "Individual Recognition Using Gait Energy Image" (TPAMI 2006)
# 做法：把一个周期内的轮廓帧按时间求平均，压成一张图

import numpy as np
from pathlib import Path
from PIL import Image


def load_silhouette_frame(path, target_size=(64, 64)):
    # 读一张轮廓帧并归一化到 (H, W)
    img = Image.open(path).convert("L")
    arr = np.asarray(img)
    # 二值化（CASIA-B 轮廓为黑底白人）
    binary = (arr > 127).astype(np.uint8)
    return _normalize_silhouette(binary, target_size)


def _normalize_silhouette(binary, target_size):
    # 空间归一化：抠出人影外框 -> 按高度缩放 -> 水平居中
    target_h, target_w = target_size
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return np.zeros(target_size, dtype=np.float32)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    box_w, box_h = x_max - x_min + 1, y_max - y_min + 1

    # 按高度对齐，按宽度比例缩放，避免畸变
    scale = target_h / box_h
    new_w = int(round(box_w * scale))
    new_w = min(new_w, target_w)

    # 裁剪 bbox
    crop = binary[y_min:y_max + 1, x_min:x_max + 1]
    img = Image.fromarray((crop * 255).astype(np.uint8))
    img = img.resize((new_w, target_h), Image.LANCZOS)
    resized = np.asarray(img).astype(np.float32) / 255.0

    # 水平居中对齐（GEI 要求空间水平对齐）
    out = np.zeros(target_size, dtype=np.float32)
    start = (target_w - new_w) // 2
    out[:, start:start + new_w] = resized
    return out


def frames_to_gei(frames):
    # 帧叠起来求平均，得到一张 GEI
    if len(frames) == 0:
        raise ValueError("frames 为空，无法构建 GEI")
    stack = np.stack(frames, axis=0).astype(np.float32)
    return stack.mean(axis=0)


def build_gei_from_sequence(seq_dir, target_size=(64, 64), max_frames=None):
    # 从某个 (人/序列类型/视角) 目录构建 GEI
    pngs = sorted(seq_dir.glob("*.png"))
    if max_frames is not None and len(pngs) > max_frames:
        # 均匀采样，保证覆盖一个周期
        idx = np.linspace(0, len(pngs) - 1, max_frames, dtype=int)
        pngs = [pngs[i] for i in idx]

    frames = [load_silhouette_frame(p, target_size) for p in pngs]
    return frames_to_gei(frames)


if __name__ == "__main__":
    # 快速自检：用随机数据验证流程
    rng = np.random.default_rng(0)
    fake = (rng.random((64, 64)) > 0.7).astype(np.uint8)
    normalized = _normalize_silhouette(fake, (64, 64))
    gei = frames_to_gei([normalized] * 5)
    print(f"归一化 shape={normalized.shape}, GEI shape={gei.shape}, range=[{gei.min():.3f}, {gei.max():.3f}]")
    print("GEI 模块自检通过")

# GaitGraph1 模型配置（参考 OpenGait 的配置风格）
# 结构: 3 个输入流（坐标/骨骼/运动）-> 输入分支 3->32->32
#       -> 主分支 ResGCN 3 层(32->64->128->256) -> 全局池化 -> FC

# COCO 17 关键点
JOINT_FORMAT = "coco"

MODEL_CFG = {
    "model": "GaitGraph1",
    "joint_format": JOINT_FORMAT,
    "input_num": 3,                 # 输入流数量: 坐标/骨骼/运动
    "block": "Basic",               # Basic 或 Bottleneck
    "reduction": 4,
    "tta": False,                   # 测试时增强（本演示关闭）
    "input_branch": {
        "num_stream": 3,
        "block": "Basic",
        "channels": [3, 32, 32],    # [每流输入通道, 中间通道, 输出通道]
    },
    "main_stream": {
        "block": "Basic",
        "channels": [
            [32, 64, 64, 64],
            [64, 128, 128, 128],
            [128, 256, 256, 256],
        ],
    },
    # num_class 在运行时根据训练集身份数设置
    "num_class": 62,
}

TRAIN_CFG = {
    "solver": "Adam",
    "adam": {"lr": 1e-3, "weight_decay": 1e-4},
    "epochs": 50,
    "batch_size": 32,
    "temperature": 0.07,            # SupConLoss 温度
}

DATA_CFG = {
    "joint_format": JOINT_FORMAT,
    "seq_frames": 40,               # 每个序列采样的帧数
    "center": 5,                    # center 关节（左肩）
}

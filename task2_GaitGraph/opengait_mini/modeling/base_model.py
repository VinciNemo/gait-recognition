# 精简版 OpenGait —— 基础模型类
# 对应 OpenGait 的 base_model.py，所有模型继承这个类

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    # 所有步态模型的基础类

    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.cfg = model_cfg
        self.build_network(model_cfg, **kwargs)

    def build_network(self, model_cfg, **kwargs):
        # 子类实现：根据配置搭网络
        raise NotImplementedError

    def forward(self, inputs):
        # 子类实现：inputs = (ipts, labs, type_, view_, seqL)
        raise NotImplementedError

    def get_optimizer(self, optim_cfg):
        # 按配置建优化器
        params = [p for p in self.parameters() if p.requires_grad]
        if optim_cfg['solver'] == 'Adam':
            return torch.optim.Adam(params, **optim_cfg['adam'])
        if optim_cfg['solver'] == 'SGD':
            return torch.optim.SGD(params, **optim_cfg['sgd'])
        raise ValueError(f"未知优化器: {optim_cfg['solver']}")

# config.py

import torch


# 参数配置
CONFIG = {
    "env_id": "LunarLander-v3",
    "seed": 11, # 用于复现性
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "total_episodes": 10000, # 总训练步数
    "sample_episodes": 5, # 每次采样的episode数
    "update_epoches": 32, # 每次更新的epoch数
    "clip_epsilon": 0.2, # PPO的剪切epsilon
    "batch_size": 32,           # 学习时的批大小
    "learning_rate": 1e-5,      # 学习率
    "gamma": 0.99,              # 折扣因子
    "gae_lambda": 0.95,         # GAE的lambda参数
    "c1":1,                     # 计算损失时对价值网络的系数
    "save_frequency": 10,     # 每多少步保存一次模型
    "model_save_dir": "PPO\model",  # 模型保存路径
    "checkpoint_path": None,     # "PPO\models/PPO_lunarlander_step_xxxxx.pth" # 可选：加载检查点继续训练
    # 环境相关
    "num_frames": 4,             # 帧堆叠数量
    "img_size": (256, 256)         # 图像预处理尺寸
}
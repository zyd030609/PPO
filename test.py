# test.py

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from datetime import datetime 
import matplotlib.pyplot as plt
import gymnasium as gym 

# 从其他模块导入
from env import ImageLunarLanderEnv # 环境包装器
from net import PG_Net     # 策略网络
from agent import PG_Agent # Agent (负责交互)
from config import CONFIG

print(f"Starting training on device: {CONFIG['device']}")
np.random.seed(CONFIG['seed'])
random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
if CONFIG['device'] == "cuda":
    torch.cuda.manual_seed(CONFIG['seed'])

device = torch.device(CONFIG['device'])

# 1. 初始化环境
env = ImageLunarLanderEnv(env_id=CONFIG['env_id'],
                            num_frames=CONFIG['num_frames'],
                            img_size=CONFIG['img_size'])
env_vis = gym.make(CONFIG['env_id'], render_mode='human')
state_shape = env.observation_space.shape # (C, H, W) -> (4, 256, 256)
num_actions = env.action_space.n # 4

# 2. 初始化网络
net = PG_Net(input_shape=state_shape, num_actions=num_actions).to(device)

# 3. 初始化 Agent 
agent = PG_Agent(state_shape=state_shape,
                    num_actions=num_actions,
                    device=device,
                    net=net,
                    if_eval=True)

# 4. 加载检查点
checkpoint = torch.load("PPO\models\PPO_lunarlander_step_10000.pth", map_location=device)
net.load_state_dict(checkpoint['net_state_dict'])
start_episode = checkpoint.get('global_episode', 0)

# 5.开始测试
torch.no_grad()
for test_episode in range(10):
    # 执行一轮的采样
    state, _ = env.reset(seed=CONFIG['seed']+test_episode) # 重置状态 (NumPy:uint8)
    env_vis.reset(seed=CONFIG['seed']+test_episode) 
    trajectory=[]
    done= False
    while(done==False):
        # 智能体根据初始状态选择动作
        action = agent.select_action(state)

        # 与环境交互
        next_state, reward, terminated, truncated, info = env.step(action)
        env_vis.step(action)
        done = terminated or truncated

        # 存储采样点
        #trajectory.append((state, action, reward, log_prob))
        trajectory.append((action, reward))       

        # 更新状态
        state = next_state

    # 计算回报，计算损失
    G=0
    loss=0
    i=0
    for _, _, reward, _, _ , _ in reversed(trajectory):
        i+=1
        G=reward+G*CONFIG['gamma']

    print(f"test_episode: {test_episode},steps:{i},G:{G}")
# train.py

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import os
from collections import deque
import random
from datetime import datetime 
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 允许重复加载OpenMP库

# 从其他模块导入
from env import ImageLunarLanderEnv # 环境包装器
from net import AC_Net     # 策略网络
from agent import PPO_Agent # Agent (负责交互)
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
state_shape = env.observation_space.shape # (C, H, W) -> (4, 256, 256)
num_actions = env.action_space.n # 4

# 2. 初始化网络
net = AC_Net(input_shape=state_shape, num_actions=num_actions).to(device)

# 3. 初始化优化器
optimizer = optim.Adam(net.parameters(), lr=CONFIG['learning_rate'])

# 4. 初始化 Agent 
agent = PPO_Agent(state_shape=state_shape,
                    num_actions=num_actions,
                    device=device,
                    net=net,
                    if_eval=False)

# 5. 初始化训练状态变量
start_episode=0
history_loss=[]
history_G=[]


# 6. 加载检查点
if CONFIG["checkpoint_path"] and os.path.exists(CONFIG["checkpoint_path"]):
    print(f"Loading checkpoint from: {CONFIG['checkpoint_path']}")
    checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device)
    net.load_state_dict(checkpoint['net_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_episode = checkpoint.get('global_episode', 0)
    history_loss = checkpoint.get('history_loss', [])
    history_G = checkpoint.get('history_G', [])
    print(f"Resuming training from episode {start_episode}")

# 7.训练循环
global_episode=start_episode
while global_episode<CONFIG['total_episodes']:
    trajectorys=[]
    for sample_episode in range(CONFIG['sample_episodes']):
        # 执行一轮的采样
        state, _ = env.reset() # 重置状态 (NumPy:uint8)
        trajectory=[]
        done= False
        while(done==False):
            # 智能体根据初始状态选择动作
            action,log_prob,v = agent.select_action(state)

            # 与环境交互
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # 存储采样点
            trajectory.append((state, action, reward, log_prob, v , done))

            # 更新状态
            state = next_state

        trajectorys.append(trajectory)
        global_episode+=1

    # 学习
    # 计算回报(真)、每episode步数，用来记录训练过程
    realGs=[]
    steps=[]
    for trajectory in trajectorys:
        G=0
        i=0
        for _, _, reward, _, _ , _ in reversed(trajectory):
            i+=1
            G=reward+G*CONFIG['gamma']
        realGs.append(float(G))
        steps.append(i)

    # 计算优势函数
    # 1.计算δ_t(TD误差)
    deltas = []
    for trajectory in trajectorys:
        trajectory_deltas = []
        for t in range(len(trajectory)):
            _, _, reward_t, _, v_t, done_t = trajectory[t]       
            # 如果是最后一步，下一个状态的价值为0
            if t == len(trajectory) - 1:
                next_v = 0
            else:
                _, _, _, _, next_v, _ = trajectory[t + 1]                
            # 计算TD误差: δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
            delta_t = reward_t + CONFIG['gamma'] * next_v - v_t
            trajectory_deltas.append(delta_t)
        deltas.append(trajectory_deltas)
    # 2.计算优势(GAE)
    As=[]
    for trajectory_deltas in deltas:
        trajectory_As=[]
        for t in reversed(range(len(trajectory_deltas))):
            if trajectory_As==[]:
                A_t=trajectory_deltas[t]
                trajectory_As.insert(0,A_t)
            else:
                A_t1=trajectory_As[0]
                delta_t=trajectory_deltas[t]
                A_t=delta_t+CONFIG['gamma']*CONFIG['gae_lambda']*A_t1
                trajectory_As.insert(0,A_t)
        As.append(trajectory_As)
    # 展平
    As_=[]
    for trajectory_As in As:
        for t in range(len(trajectory_As)):
            As_.append(trajectory_As[t])
    trajectorys_=[]
    for trajectory in trajectorys:
        for t in range(len(trajectory)):
            trajectorys_.append(trajectory[t])
    # 对As_进行标准化
    As_ = torch.stack(As_)
    As_ = (As_ - As_.mean()) / (As_.std() + 1e-8)
    # 3.计算回报（G）
    Gs = []
    for t in range(len(trajectorys_)):
        _, _, _, _, v_t, _ = trajectorys_[t]
        A_t = As_[t]
        G_t = A_t + v_t
        Gs.append(G_t)
    # 4.合并所有轨迹的采样点
    samples = []
    for i in range(len(trajectorys_)):
        state_t, action_t, reward_t, log_prob_t, v_t, done_t = trajectorys_[i]
        A_t = As_[i]
        G_t = Gs[i]
        samples.append((state_t, action_t, reward_t, log_prob_t, v_t, done_t, A_t, G_t))

    # 计算损失并更新网络
    losses_=[]
    for epoch in range(CONFIG['update_epoches']): #多轮迭代更新
        # 计算损失（每次更新都要单独计算）
        losses=[]
        batch = random.sample(samples, CONFIG["batch_size"]) #采样一个batch，每轮只采样一个batch
        for state_t, action_t, _, log_prob_old_t, _, _, A_t, G_t in batch:
            # 重新计算新策略下该状态的 log_prob
            log_prob_new_t, v_new_t = agent.evaluate_action(state_t, action_t)
            ratio_t = torch.exp(log_prob_new_t - log_prob_old_t)
            loss_A_1 = ratio_t * A_t
            loss_A_2 = torch.clamp(ratio_t, 1 - CONFIG['clip_epsilon'], 1 + CONFIG['clip_epsilon']) * A_t
            loss_A = -torch.min(loss_A_1, loss_A_2)
            loss_C = F.mse_loss(v_new_t, torch.tensor(G_t, dtype=torch.float32, device=device))
            loss=loss_A + CONFIG['c1'] * loss_C
            losses.append(loss)
        loss = torch.stack(losses).mean() #取平均
        losses_.append(loss.item())
        # 反向传播，更新策略网络
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    ave_loss = np.mean(losses_)
    print(f"global_episode: {global_episode}, in {CONFIG['sample_episodes']} episodes, steps: {steps}, G: {[round(g, 1) for g in realGs]}, ave_loss: {round(ave_loss, 1)}")
    #print(trajectory)

    # 记录训练过程变量
    history_loss.append(ave_loss)
    for i in realGs:
        history_G.append(i)

    # 定期保存模型
    if global_episode % CONFIG['save_frequency'] == 0:
        save_dir = CONFIG['model_save_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"PPO_lunarlander_step_{global_episode}.pth")
        checkpoint_data = {
            'global_episode': global_episode,
            'net_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history_loss':history_loss,
            'history_G':history_G
        }
        torch.save(checkpoint_data, save_path)
        print(f"\nCheckpoint saved to {save_path}")

def visualize(history_g, history_loss):
    history_g_np = np.array(history_g)   
    history_loss_np = np.array(history_loss) 
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history_g_np, label='Episodic Return (G)', alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('Return (G)')
    plt.title('Episodic Return over Training')
    plt.legend()
    plt.grid(True)
   
    plt.subplot(1, 2, 2)
    plt.plot(history_loss_np, label='Loss', color='green', alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss over Training')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

visualize(history_G,history_loss)


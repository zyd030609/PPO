# visualize.py

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
from config import CONFIG 

device = torch.device(CONFIG['device'])

def moving_average(data, window_size):
    """ 计算一维数据的滑动平均值 """
    # 使用卷积计算滑动平均
    # 'valid'模式确保输出的每个点都是由完整的窗口计算得出的
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

checkpoint = torch.load("PPO\model\PPO_lunarlander_step_10.pth", map_location=device)
start_episode = checkpoint.get('global_episode', 0)
history_loss = checkpoint.get('history_loss', [])
history_G = checkpoint.get('history_G', [])

# --- Visualize ---
def visualize(history_g, history_loss, moving_avg_window=100):
    history_g_np = np.array(history_g)
    history_loss_np = np.array(history_loss)
    plt.figure(figsize=(15, 6)) # Increased figure width for better readability

    plt.subplot(1, 2, 1)
    plt.plot(history_g_np, label='Episodic Return (G)', alpha=0.5, color='skyblue') # Made original plot lighter

    # Calculate and plot moving average for G

    avg_G = moving_average(history_g_np, moving_avg_window)
    plt.plot(np.arange(moving_avg_window - 1, len(history_g_np)), avg_G,
                label=f'Moving Avg Return (Window {moving_avg_window})', color='dodgerblue', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Return (G)')
    plt.title('Episodic Return over Training')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history_loss_np, label='Loss', color='mediumseagreen', alpha=0.7) 
    avg_loss = moving_average(history_loss_np, moving_avg_window)
    plt.plot(np.arange(moving_avg_window - 1, len(history_loss_np)), avg_loss,
                label=f'Moving Avg Loss (Window {moving_avg_window})', color='darkgreen', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss over Training')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

visualize(history_G, history_loss, moving_avg_window=100) 

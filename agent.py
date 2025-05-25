# agent.py 

import torch
import torch.nn as nn
import torch.distributions as D
import numpy as np
import random
from typing import Tuple

from net import AC_Net

class PPO_Agent:
    """
    PPO 智能体，只负责动作选择。
    """

    def __init__(self,
                 state_shape,
                 num_actions: int,
                 device: torch.device,
                 net: nn.Module,
                 if_eval = False): 
        """
        初始化 PPO_Agent。

        Args:
            state_shape (Tuple[int, int, int]): 输入状态的形状 (frames, height, width)。
            num_actions (int): 可执行动作的数量。
            device (torch.device): 计算设备 ('cpu' or 'cuda')。
            net (nn.Module): 用于决策的策略网络实例。
            if_eval = False:训练时使用;if_eval = True:测试时使用。
        """
        self.state_shape=state_shape
        self.num_actions = num_actions
        self.device = device
        self.net = net
        self.if_eval=if_eval

        if self.if_eval:
            self.net.eval()
        else:
            self.net.train()
        
        print(f"PPO Agent initialized with {num_actions} actions on device {device}.")

    def select_action(self, state_np: np.ndarray) -> int:
        """
        将当前状态输入策略网络，根据输出概率随机选择一个动作。

        Args:
            state_np (np.ndarray): 当前环境状态 (预处理、堆叠后的 NumPy 数组, dtype=uint8)。

        Returns:
            int: 选择的动作索引。
            tensor: 动作的 log_prob (如果不是评估模式)。
            tensor: 状态价值 V(s) (如果不是评估模式)。
        """
        with torch.no_grad():
            state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device).float() / 255.0
            action_logits,v = self.net(state_tensor) # Shape: (1, num_actions)
            if not self.if_eval:
                action_distribution = D.Categorical(logits=action_logits)
                action_tensor = action_distribution.sample()
                log_prob = action_distribution.log_prob(action_tensor)
                return action_tensor.item(), log_prob.squeeze(),v.squeeze() 
            else:
                action = action_logits.argmax(dim=1).item()
                return action
        
    def evaluate_action(self, state_t_np: np.ndarray,action_t: int):
        """
        输入状态和所选动作，计算该动作的log概率及该状态的价值v，用于计算损失。

        Args:
            state_t_np (np.ndarray): 当前环境状态 (预处理、堆叠后的 NumPy 数组, dtype=uint8)。
            action_t (int): 所选动作。

        Returns:
            tensor: 动作的 log_prob 。
            tensor: 状态价值 V(s) 。
        """
        state_t_tensor = torch.from_numpy(state_t_np).unsqueeze(0).to(self.device).float() / 255.0
        action_t_logits,v_new = self.net(state_t_tensor) # Shape: (1, num_actions)

        action_t_distribution = D.Categorical(logits=action_t_logits)
        action_t_tensor = torch.tensor([action_t], device=self.device)
        log_prob_new = action_t_distribution.log_prob(action_t_tensor)
        return log_prob_new.squeeze(),v_new.squeeze() 
    
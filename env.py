# env.py
"""
封装了自己的环境。主要目的是将环境返回的状态改为图像序列。
"""

import gymnasium as gym
import numpy as np
import cv2  
from collections import deque

class ImageLunarLanderEnv:
    """
    一个包装器，用于LunarLander-v2环境，强制使用预处理后的图像作为状态。
    处理包括：获取渲染图像、灰度化、缩放、帧堆叠。
    """
    def __init__(self, env_id='LunarLander-v3', num_frames=4, img_size=(84, 84),render_mode_env='rgb_array'):
        """
        初始化环境和处理参数。

        Args:
            env_id (str): Gymnasium环境ID。
            num_frames (int): 要堆叠的帧数。
            img_size (tuple): 预处理后图像的目标尺寸 (height, width)。
        """
        # 创建基础环境，指定渲染模式为'rgb_array'以获取图像
        self.env = gym.make(env_id, render_mode=render_mode_env)

        self.num_frames = num_frames
        self.img_size = img_size
        self.frame_stack = deque(maxlen=num_frames)

        # 获取原始动作空间
        self.action_space = self.env.action_space

        # 定义新的观察空间（状态空间）
        # 形状是 (num_frames, height, width) 
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=(self.num_frames, self.img_size[0], self.img_size[1]),  #4，84，84
            dtype=np.uint8
        )
        print(f"Wrapped Env Observation Space: {self.observation_space.shape}")
        print(f"Original Env Action Space: {self.action_space.n}")


    def _preprocess_frame(self, frame):
        """
        对单帧图像进行预处理：灰度化和缩放。

        Args:
            frame (np.ndarray): 原始RGB图像 (H, W, C)。

        Returns:
            np.ndarray: 预处理后的图像 (img_size[0], img_size[1])，dtype=uint8。
        """
        if frame is None:
             # 处理 render() 可能返回 None 的情况 (虽然不常见)
             print("Warning: Received None frame during preprocessing.")
             # 返回一个全黑的帧或者上一个有效帧？这里返回全黑
             return np.zeros(self.img_size, dtype=np.uint8)
             
        # 1. 转换为灰度图
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # 2. 缩放图像
        resized_frame = cv2.resize(gray_frame, (self.img_size[1], self.img_size[0]), # cv2 resize dsize is (width, height)
                                   interpolation=cv2.INTER_AREA) #插值方法

        return resized_frame.astype(np.uint8) # 确保是 uint8

    def _get_stacked_frames(self):
        """
        从deque中获取帧并堆叠成一个numpy数组。
        """
        assert len(self.frame_stack) == self.num_frames, "Frame stack size incorrect!"
        # 按照 Channels-first (num_frames, height, width) 堆叠
        return np.stack(list(self.frame_stack), axis=0)

    def reset(self, seed=None, options=None):
        """
        重置环境，获取初始状态（堆叠帧）。

        Returns:
            np.ndarray: 初始的堆叠帧状态 (num_frames, H, W)。
            dict: 附加信息 (info)。
        """
        # 重置底层环境，获取 info
        _, info = self.env.reset(seed=seed, options=options)

        # 获取初始渲染帧
        initial_frame = self.env.render()
        if initial_frame is None:
             raise RuntimeError("Environment failed to render on reset.")

        # 将rgb图像转换为灰度图像     
        processed_frame = self._preprocess_frame(initial_frame)

        # 用初始帧填满整个帧堆叠队列
        self.frame_stack.clear()
        for _ in range(self.num_frames):
            self.frame_stack.append(processed_frame)

        # 获取堆叠后的初始状态。返回一个ndarray
        stacked_state = self._get_stacked_frames()

        return stacked_state, info

    def step(self, action):
        """
        在环境中执行一步，获取下一个状态、奖励等。

        Args:
            action: 要执行的动作。

        Returns:
            tuple: (next_stacked_state, reward, terminated, truncated, info)
                   next_stacked_state (np.ndarray): 下一个堆叠帧状态。
                   reward (float): 奖励。
                   terminated (bool): 环境是否终止 (成功或失败)。
                   truncated (bool): 环境是否截断 (如达到时间限制)。
                   info (dict): 附加信息。
        """
        # 执行动作，获取原始信息
        # 不直接用返回的 observation
        _, reward, terminated, truncated, info = self.env.step(action)

        # 获取新的渲染帧
        next_frame = self.env.render()
        # if next_frame is None:
        #      # 如果渲染失败，可能需要决定如何处理
        #      # 例如，重用上一帧？或认为 episode 结束？
        #      # 这里我们假设渲染总是成功的，或者如果失败则是一个严重问题
        #      # 或者我们可以让 truncated = True?
        #      print("Warning: Frame rendering failed during step. Using previous frame stack.")
        #      # 返回之前的状态可能导致学习问题，这里或许应该抛出错误或处理
        #      # 简单起见，我们先保持原样，但在实际应用中要小心
        #      next_stacked_state = self._get_stacked_frames() 
        # else:
        processed_frame = self._preprocess_frame(next_frame)

        # 将新帧添加到队列中
        self.frame_stack.append(processed_frame)

        # 获取新的堆叠状态
        next_stacked_state = self._get_stacked_frames()

        return next_stacked_state, reward, terminated, truncated, info

    def render(self, mode='human'):
        """
        提供一个渲染接口，用于可视化（如果需要的话）。
        注意：如果原始环境是用 'rgb_array' 初始化的，直接调用 render() 可能不会弹出窗口。
        可能需要创建一个临时的 'human' 模式环境来显示。
        或者直接使用 agent 训练/测试循环中获取的 frame 进行显示 (e.g., using cv2.imshow)。
        简单起见，这里只返回 rgb_array 供外部显示。
        """
        frame = self.env.render()
        # 如果想在调用此方法时看到窗口，需要更复杂的处理
        # 例如，单独创建一个 'human' mode 的环境实例
        return frame # 返回原始渲染帧，方便外部显示

    def close(self):
        """
        关闭环境。
        """
        self.env.close()


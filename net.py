import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

class AC_Net(nn.Module):
    def __init__(self, input_shape, num_actions):
        """
        Actor-Critic 网络，使用CNN处理图像输入。

        参数:
        input_channels (int): 输入图像的通道数 (例如，堆叠的帧数，如 4)。
        num_actions (int): 离散动作空间的大小 (例如，LunarLander为4)。
        """
        super(AC_Net, self).__init__()
        input_channels = input_shape[0] # 4
        height = input_shape[1]     # 256
        width = input_shape[2]      # 256
        self.num_actions = num_actions

        # 卷积层
        # 输入形状: (batch_size, input_channels, H, W)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=8, stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        # 根据 input_shape 计算展平后的特征维度
        conv_out_size = self._get_conv_output_size((input_channels, height, width))
        print(f"计算得到的卷积层输出大小: {conv_out_size}")

        # Actor Head (策略网络)
        self.actor_head = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions) # 输出层 (每个动作分数)
        )

        # Critic Head (价值网络)
        self.critic_head = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1) # 输出状态价值V(s)
        )

        # 权重初始化 (可选，但有时有帮助)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        初始化网络权重。
        """
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
            # nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _get_conv_output_size(self, shape):
        """
        辅助函数，用于计算卷积层输出展平后的大小。
        shape: (channels, height, width)
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape) # 创建一个 batch_size=1 的虚拟输入
            output = self.conv_layers(dummy_input)
            return int(np.prod(output.size()[1:])) # 忽略batch维度，计算剩余维度乘积

    def forward(self, x):
        """
        前向传播。

        参数:
        x (torch.Tensor): 输入的图像数据，形状 (batch_size, input_channels, H, W)。
                          确保像素值已归一化 (例如，到 [0, 1] 或 [-1, 1])。

        返回:
        action_logits (torch.Tensor): 动作的 logits，形状 (batch_size, num_actions)。
        v (torch.Tensor): 状态价值，形状 (batch_size, 1)。
        """
        # 归一化输入图像 (如果尚未在预处理中完成)
        # x = x / 255.0 # 假设输入是 [0, 255] 的 uint8

        # 通过卷积层提取特征
        features = self.conv_layers(x) # (batch_size, flattened_size)

        # Actor head: 计算动作 logits
        action_logits = self.actor_head(features) # (batch_size, num_actions)

        # Critic head: 计算状态价值
        v = self.critic_head(features) # (batch_size, 1)

        return action_logits, v


if __name__ == '__main__':
    input_shape = (4, 256, 256)
    num_actions = 4
    model = AC_Net(input_shape, num_actions)
    dummy_input = torch.randn(1, *input_shape)
    output = model(dummy_input)


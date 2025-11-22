import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Sequential, Linear, ReLU, ModuleList, Softmax, ELU, Sigmoid
from module.utils import rl_utils  # 注意：保持原来导入方式
import torch
import torch.nn.functional as F
import gymnasium as gym

# 定义策略网络（PolicyNet），用于生成动作分布
class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 输入层到隐藏层
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)  # 隐藏层到输出层
    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用 ReLU 激活函数
        return F.softmax(self.fc2(x), dim=1)  # 输出动作概率分布，使用 softmax 激活
# 定义价值网络（ValueNet），用于评估状态的价值
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)  # 输入层到隐藏层
        self.fc2 = torch.nn.Linear(hidden_dim, 1)  # 隐藏层到输出层
    def forward(self, x):
        x = F.relu(self.fc1(x))  # 使用 ReLU 激活函数
        return self.fc2(x)  # 输出状态价值
# 定义 PPO 算法，采用截断（Clipping）方式
class PPO:
    ''' PPO 算法,采用截断方式 '''
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)  # 策略网络
        self.critic = ValueNet(state_dim, hidden_dim).to(device)  # 价值网络
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)  # 策略网络优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)  # 价值网络优化器
        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE 参数
        self.epochs = epochs  # 每次更新训练的轮数
        self.eps = eps  # PPO 中截断范围参数
        self.device = device  # 使用的设备（CPU 或 GPU）
    def take_action(self, state):
        # 根据当前策略网络对状态 state 进行采样，生成一个动作
        state = torch.tensor([state], dtype=torch.float).to(self.device)  # 转换为张量并传输到设备
        probs = self.actor(state)  # 获取动作概率分布
        action_dist = torch.distributions.Categorical(probs)  # 定义一个类别分布
        action = action_dist.sample()  # 从类别分布中采样一个动作
        return action.item()  # 返回动作（整数）
    def update(self, transition_dict):
        # 更新策略和价值网络
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)  # 状态
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)  # 动作
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)  # 奖励
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)  # 下一状态
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)  # 是否结束
        # 计算 TD 目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)  # TD 残差
        # 使用工具函数计算优势函数
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(self.device)
        # 计算旧策略下的动作对数概率
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))  # 重新计算对数概率
            ratio = torch.exp(log_probs - old_log_probs)  # 比例因子 r_theta
            surr1 = ratio * advantage  # 未裁剪项
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断项
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # 策略损失，因为我们要最大化策略目标，所以取负号，将其转换为损失函数（梯度下降算法需要最小化损失）
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))  # 价值网络损失
            # 更新参数
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()  # 反向传播更新策略网络
            critic_loss.backward()  # 反向传播更新价值网络
            self.actor_optimizer.step()
            self.critic_optimizer.step()



if __name__ == "__main__":
    print(1)
    # 超参数设置
    actor_lr = 1e-3  # 策略网络学习率
    critic_lr = 1e-2  # 价值网络学习率
    num_episodes = 500  # 训练的总回合数
    hidden_dim = 128  # 隐藏层维度
    gamma = 0.98  # 折扣因子
    lmbda = 0.95  # GAE 参数
    epochs = 10  # 每次更新的轮数
    eps = 0.2  # PPO 截断范围
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 设备设置（优先使用 GPU）
    # 创建环境
    env_name = 'CartPole-v0'  # 环境名称
    env = gym.make(env_name)  # 创建 Gym 环境
    env.reset(seed=0)  # 设置随机种子
    torch.manual_seed(0)  # 设置 PyTorch 随机种子
    state_dim = env.observation_space.shape[0]  # 状态空间维度
    action_dim = env.action_space.n  # 动作空间维度
    agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)  # 初始化 PPO 算法
    # 开始训练
    return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)  # 使用工具函数训练
    episodes_list = list(range(len(return_list)))  # 生成回合序列
    plt.plot(episodes_list, return_list)  # 绘制回报曲线
    plt.xlabel('Episodes')  # x 轴标签
    plt.ylabel('Returns')  # y 轴标签
    plt.title('PPO on {}'.format(env_name))  # 标题
    plt.show()  # 显示图像
    # 绘制平滑曲线
    # mv_return = rl_utils.moving_average(return_list, 9)  # 计算滑动平均回报
    # plt.plot(episodes_list, mv_return)  # 绘制滑动平均曲线
    # plt.xlabel('Episodes')  # x 轴标签
    # plt.ylabel('Returns')  # y 轴标签
    # plt.title('PPO on {}'.format(env_name))  # 标题
    # plt.show()  # 显示图像
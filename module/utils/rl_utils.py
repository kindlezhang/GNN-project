from tqdm import tqdm
import gymnasium as gym  # 导入 Gymnasium 环境库，用于模拟强化学习环境
import torch  # 导入 PyTorch，作为深度学习的核心框架
import torch.nn.functional as F  # 导入 PyTorch 的常用函数库，包括激活函数、损失函数等
import numpy as np  # 导入 NumPy，用于数值计算
import matplotlib.pyplot as plt  # 导入 Matplotlib，用于绘图


def train_on_policy_agent(env, agent, num_episodes):
    return_list = []  # 存储每回合的总回报
    for i in range(10):  # 将训练分为 10 个阶段
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:  # 创建进度条
            for i_episode in range(int(num_episodes/10)):  # 每阶段训练 num_episodes/10 回合
                episode_return = 0  # 初始化当前回合的总回报
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}  # 记录当前回合的数据
                state, info = env.reset()  # 重置环境
                done = False
                while not done:  # 游戏未结束时继续
                    action = agent.take_action(state)  # 选择动作
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    transition_dict['states'].append(state)  # 记录状态
                    transition_dict['actions'].append(action)  # 记录动作
                    transition_dict['next_states'].append(next_state)  # 记录下一状态
                    transition_dict['rewards'].append(reward)  # 记录奖励
                    transition_dict['dones'].append(done)  # 记录是否结束
                    state = next_state  # 更新当前状态
                    episode_return += reward  # 累加回报
                return_list.append(episode_return)  # 记录当前回合总回报
                agent.update(transition_dict)  # 使用当前回合数据更新策略
                if (i_episode+1) % 10 == 0:  # 每 10 个回合更新进度条信息
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)  # 更新进度条
    return return_list  # 返回所有回合的总回报
    
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()  # 将 TD-误差转换为 NumPy 数组
    advantage_list = []  # 初始化优势值列表
    advantage = 0.0  # 初始化递归变量
    for delta in td_delta[::-1]:  # 从后往前遍历 TD-误差
        advantage = gamma * lmbda * advantage + delta  # 递归计算优势值
        advantage_list.append(advantage)  # 存储优势值
    advantage_list.reverse()  # 恢复正序
    return torch.tensor(np.array(advantage_list), dtype=torch.float)  # 返回优势值张量
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Sequential, Linear, ReLU, ModuleList, Softmax, ELU, Sigmoid
from module.utils import rl_utils  # 注意：保持原来导入方式
import torch
import torch.nn.functional as F
import gymnasium as gym
from module.utils.reorganizer import relabel_graph, filter_correct_data, filter_correct_data_batch
from torch_geometric.utils import softmax


# device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# def group_softmax(scores: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
#     """
#     类似 torch_geometric.utils.softmax(scores, batch_index)，
#     但完全用 PyTorch 实现，兼容 MPS。
    
#     scores: [num_edges] 或 [num_edges, 1]
#     batch_index: [num_edges]，每个元素表示该边属于哪个图
#     """
#     if scores.dim() == 2 and scores.size(1) == 1:
#         scores = scores.view(-1)

#     probs = torch.zeros_like(scores)
#     unique_batches = batch_index.unique()

#     for b in unique_batches:
#         mask = (batch_index == b)
#         # 对同一个图里的所有候选边做 softmax
#         probs[mask] = F.softmax(scores[mask], dim=0)

#     return probs


class PolicyNet(nn.Module):
    def __init__(self, gnn_model, num_labels, hidden_size, use_edge_attr=False):
        super(PolicyNet, self).__init__()

        # 1. 引入预训练的 GNN 模型用于提取特征
        self.model = gnn_model
        self.model.eval() # 固定 GNN 参数，只训练 PolicyNet
        
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.use_edge_attr = use_edge_attr

        # 2. 边特征提取器 (对应原代码 edge_action_rep_generator)
        # 输入维度: 2个节点特征 + (可选)边特征
        input_dim = self.hidden_size * 2
        if self.use_edge_attr:
            input_dim += self.hidden_size # 假设边嵌入维度也是 hidden_size
            
        self.edge_action_rep_generator = Sequential(
            Linear(input_dim, self.hidden_size * 2),
            ELU(),
            Linear(self.hidden_size * 2, self.hidden_size),
            ELU(),
            Linear(self.hidden_size, self.hidden_size)
        ).to(device)

        # 3. 动作概率生成器 (对应原代码 build_edge_action_prob_generator)
        # 针对每个类别(label)有一个独立的评分网络
        self.edge_action_prob_generators = nn.ModuleList()
        for _ in range(self.num_labels):
            i_explainer = Sequential(
                # 输入包含: 边本身的 embedding + (全图表示 - 子图表示)
                Linear(self.hidden_size * 2, self.hidden_size * 2),
                ELU(),
                Linear(self.hidden_size * 2, self.hidden_size),
                ELU(),
                Linear(self.hidden_size, 1)
            ).to(device)
            self.edge_action_prob_generators.append(i_explainer)

    def forward(self, graph, state):
        """
        Args:
            graph: PyG 的 Batch Data 对象
            state: Boolean Tensor, 形状为 [num_total_edges], True 表示该边已在子图中
        Returns:
            probs: 可选边的概率分布
            ava_action_batch: 可选边所属的 graph batch index (用于后续采样或处理)
        """
        
        # 1. 获取全图的 Graph Representation
        # graph_rep shape: [batch_size, hidden_size]
        with torch.no_grad():
            graph_rep = self.model.get_graph_rep(graph.x, graph.edge_index, graph.edge_attr, graph.batch)

        # 2. 获取当前子图 (由 state 定义) 的 Graph Representation
        # 如果 state 全为 False (初始状态)，子图表示为 0
        if state.sum() == 0:
            subgraph_rep = torch.zeros_like(graph_rep).to(device)
        else:
            # 注意：需要你需要保证 relabel_graph 函数可用
            # 这里通常是从 utils 导入的函数，用于根据 mask 创建新 graph 对象
            subgraph = relabel_graph(graph, state)
            with torch.no_grad():
                subgraph_rep = self.model.get_graph_rep(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)

        # 3. 确定“可选动作” (Available Actions)
        # 动作空间定义为：还未被选入子图的边 (~state)
        ava_mask = ~state
        ava_edge_index = graph.edge_index[:, ava_mask]
        ava_edge_attr = graph.edge_attr[ava_mask] if graph.edge_attr is not None else None
        
        # 获取可选边对应的 Batch Index 和 Target Label
        ava_action_batch = graph.batch[ava_edge_index[0]] # 知道每条边属于 batch 里的哪个图
        ava_y_batch = graph.y[ava_action_batch] # 获取每条边对应图的标签

        # 4. 计算动作(边)的基础特征 (Node u + Node v + Edge Attr)
        with torch.no_grad():
            # 获取所有节点的表示
            all_node_reps = self.model.get_node_reps(graph.x, graph.edge_index, graph.edge_attr, graph.batch)
        
        # 拼接源节点和目标节点的特征
        row, col = ava_edge_index
        ava_action_reps = torch.cat([all_node_reps[row], all_node_reps[col]], dim=1)

        if self.use_edge_attr and ava_edge_attr is not None:
            with torch.no_grad():
                edge_emb = self.model.edge_emb(ava_edge_attr)
            ava_action_reps = torch.cat([ava_action_reps, edge_emb], dim=1)

        # 通过 MLP 压缩特征
        ava_action_reps = self.edge_action_rep_generator(ava_action_reps) # Shape: [num_ava_edges, hidden_size]

        # 5. 结合上下文信息 (Global - Local)
        # 处理 Batch 对齐问题：有的图可能边选完了，或者 batch 索引不连续
        # 使用 unique 确保索引对齐
        unique_batch_ids, inverse_indices = torch.unique(ava_action_batch, return_inverse=True)
        
        # 扩展 subgraph_rep 以匹配当前处理的 unique batch
        # 注意：这里需要小心处理，确保维度匹配。
        # 简单做法：直接用 ava_action_batch 索引去取 graph_rep 和 subgraph_rep
        
        current_graph_rep = graph_rep[ava_action_batch]
        current_subgraph_rep = subgraph_rep[ava_action_batch]
        
        # 核心逻辑：利用 全图 和 当前子图 的差异作为上下文
        context_diff = current_graph_rep - current_subgraph_rep
        
        # 拼接：[边特征, 图差异特征]
        full_action_reps = torch.cat([ava_action_reps, context_diff], dim=1) # Shape: [num_ava_edges, hidden_size * 2]

        # 6. 预测概率 (Predict)
        # 由于每个图的 label 可能不同，我们需要根据 label 选择对应的 MLP 头
        # 这里为了并行计算，我们将所有数据过一遍对应的 MLP，然后 gather
        
        # 这种写法比循环 list 更快，前提是显存够用。如果显存紧张，可以用循环。
        # 这里沿用原本逻辑，分别计算所有类别的分数，然后 gather
        all_scores = []
        for explainer_head in self.edge_action_prob_generators:
            score = explainer_head(full_action_reps)
            all_scores.append(score)
        
        all_scores = torch.cat(all_scores, dim=1) # [num_ava_edges, num_labels]
        
        # 根据每个图真实的 label 取出对应的分数
        target_scores = all_scores.gather(1, ava_y_batch.view(-1, 1)).view(-1)
        
        # 7. Softmax 生成概率
        # 使用 torch_geometric 的 softmax，它会在每个 graph (batch) 内部进行归一化
        probs = softmax(target_scores, ava_action_batch)
        
        # 返回概率和对应的 batch 索引（方便外部知道这些概率属于哪些图）
        return probs, ava_action_batch

class GraphValueNet(nn.Module):
    def __init__(self, gnn_model, hidden_size):
        super(GraphValueNet, self).__init__()
        
        # 1. 复用 GNN 模型提取特征 (共享特征提取层通常能加速收敛)
        self.model = gnn_model
        self.model.eval() # 固定 GNN 参数
        
        self.hidden_size = hidden_size

        # 2. 价值评估网络 (Critic Head)
        # 输入: 全图 Embedding + 子图 Embedding (或者它们的差值/拼接)
        # 输出: 1个标量 (Value)
        self.value_head = Sequential(
            Linear(self.hidden_size * 2, self.hidden_size), # 假设拼接
            ELU(),
            Linear(self.hidden_size, self.hidden_size // 2),
            ELU(),
            Linear(self.hidden_size // 2, 1) # 输出状态价值 V(s)
        ).to(device)

    def forward(self, graph, state):
        """
        Args:
            graph: PyG Batch 对象
            state: [num_total_edges] 的 bool mask
        Returns:
            values: [batch_size, 1] 每个图的状态价值
        """
        # 1. 获取全图表示 [batch_size, hidden_dim]
        with torch.no_grad():
            graph_rep = self.model.get_graph_rep(graph.x, graph.edge_index, graph.edge_attr, graph.batch)

        # 2. 获取子图表示 [batch_size, hidden_dim]
        if state.sum() == 0:
            subgraph_rep = torch.zeros_like(graph_rep).to(device)
        else:
            # 使用 mask 构建子图
            subgraph = relabel_graph(graph, state)
            with torch.no_grad():
                subgraph_rep = self.model.get_graph_rep(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)

        # 3. 处理 Batch 对齐问题
        # 在 PyG 中，relabel_graph 后 subgraph.batch 的顺序通常与 graph.batch 是一致的（或者是其子集）。
        # 但为了保险，以及处理某些图可能在子图中完全没有边的情况（relabel_graph可能会丢弃节点），
        # 最稳健的做法是根据 batch index 对齐。
        
        # 简单假设：relabel_graph 保持了 batch 的完整性（即 batch size 不变，只是某些图特征变零）
        # 如果 relabel_graph 返回的 batch size 可能会变小，这里需要额外的 scatter/gather 处理。
        # 这里假设你的 utils 里的 get_graph_rep 能够处理 batch 并在空图时返回 0 向量。

        # 4. 拼接特征
        # 状态 = (目标是什么, 现在做到了多少) -> (全图, 子图)
        state_rep = torch.cat([graph_rep, subgraph_rep], dim=1) # [batch_size, hidden * 2]

        # 5. 预测价值
        value = self.value_head(state_rep) # [batch_size, 1]
        
        return value
    
def calculate_reward(gnn_model, graph_batch, current_mask, alpha=1.0):
    """
    计算奖励：子图预测分布与全图预测分布的负KL散度
    """
    gnn_model.eval()
    with torch.no_grad():
        # 1. 全图预测 (Target)
        full_logits = gnn_model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr, graph_batch.batch)
        full_probs = F.softmax(full_logits, dim=1)
        
        # 2. 子图预测 (Current)
        # 使用 mask 构建子图
        subgraph = relabel_graph(graph_batch, current_mask)
        sub_logits = gnn_model(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)
        sub_probs = F.softmax(sub_logits, dim=1)
        
        # 3. 计算 KL 散度: sum( p_full * log(p_full / p_sub) )
        # 注意：为了数值稳定，加上 1e-8
        kl = F.kl_div((sub_probs + 1e-8).log(), full_probs, reduction='none').sum(dim=1)
        
        # 4. 奖励：KL 越小越好，所以取负数 (或者 exp(-kl))
        # 可以在这里加入稀疏度惩罚： reward = -kl - lambda * num_edges
        rewards = -kl * alpha
        
    return rewards.view(-1, 1) # [batch_size, 1]


# 定义 PPO 算法，采用截断（Clipping）方式
class PPO_Graph:
    def __init__(self, gnn_model, num_labels, hidden_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.device = device
        self.gnn_model = gnn_model
        
        # 1. 修改初始化，匹配 Graph PolicyNet 定义
        self.actor = PolicyNet(gnn_model, num_labels, hidden_dim).to(device)
        self.critic = GraphValueNet(gnn_model, hidden_dim).to(device)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps

    def take_action(self, graph, state):
        """
        Args:
            graph: PyG Batch
            state: Boolean Mask [num_edges]
        """
        # 1. 获取概率分布
        # probs shape: [num_available_edges]
        # batch_index shape: [num_available_edges] (指示每条边属于哪个图)
        probs, batch_index = self.actor(graph, state)
        
        # 2. 按图进行采样 (Per-Graph Sampling)
        # 这是一个难点。简单的 Categorical 不能处理 Batch 大小不一的情况。
        # 方案：我们这里手动实现基于 Batch 的采样，或者利用 Gumbel-Max Trick
        # 为了简单稳定，我们循环处理（虽然慢一点，但逻辑清晰），或者使用 mask 技巧。
        
        # 这里使用一个简单的 Trick：由于 probs 已经是 softmax 过的（在 batch 内和为1）
        # 我们可以直接用 torch.multinomial 吗？不行，因为 probs 是拼在一起的。
        
        # === 修正后的采样逻辑 ===
        action_indices = []
        
        # 获取 batch 中唯一的图 ID
        unique_graphs = torch.unique(batch_index)
        
        # 警告：这种循环在 Python 层面可能会慢，如果 Batch 很大需要优化为向量化操作
        for graph_id in unique_graphs:
            # 找到属于当前图的所有边的 mask
            mask = (batch_index == graph_id)
            
            # 取出该图的概率分布
            graph_probs = probs[mask]
            
            # 在该图内采样
            dist = torch.distributions.Categorical(graph_probs)
            action_local_idx = dist.sample()
            
            # 将局部索引转换为全局 probs 中的索引
            # 找到 mask 为 True 的位置的索引
            global_indices = torch.nonzero(mask).squeeze(1)
            action_global_idx = global_indices[action_local_idx]
            
            action_indices.append(action_global_idx)
            
        # 返回选中的边的全局索引（在 ava_edge_index 中的索引，不是 graph.edge_index）
        return torch.stack(action_indices) 

    def update(self, transition_dict, graph_batch):
        """
        特别注意：Graph PPO 的 update 比较复杂，因为还要传入 graph_batch
        Args:
            transition_dict: 包含 'states' (masks), 'actions', 'rewards' 等
            graph_batch: 对应的 PyG Batch 对象 (必须和 transition_dict 里的数据对应)
        """
        # 整理数据
        # states 是 mask 的列表，我们需要把它堆叠起来或者保持原样，取决于你的 forward 实现
        # 但在这里，因为 graph 没变，我们只需要处理 mask
        
        # 注意：为了简化，这里假设 transition_dict 存储的是 One-Step 或者 One-Batch 的数据
        # 如果是多步 Trajectory，需要小心 graph_batch 是否匹配
        
        states = torch.stack(transition_dict['states']).to(self.device) # [seq_len, num_edges] ? 
        # 如果 seq_len > 1，这里处理起来非常麻烦，因为 forward 一次只能接受一个 mask
        # 建议：先只做单步更新 (One-step update) 或者把 mask 拆开循环 update
        
        actions = torch.cat(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.cat(transition_dict['rewards']).view(-1, 1).to(self.device)
        next_states = torch.stack(transition_dict['next_states']).to(self.device)
        if len(transition_dict['dones']) > 0 and isinstance(transition_dict['dones'][0], torch.Tensor):
            dones = torch.cat(transition_dict['dones']).view(-1, 1).to(self.device)
        else:
            # 如果是普通的 python bool/int 列表，先转 tensor
            dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # === 计算 TD Target ===
        # 1. 计算 TD Target 和 Advantage
        td_target_list = []
        td_delta_list = []
        
        # 我们需要遍历时间步 (T)
        # 注意：actions/rewards 已经是 cat 过的长条了，但在计算 value 时我们需要按 step 拆分
        # 或者直接循环 transition_dict 的原始 list
        
        for i in range(len(transition_dict['states'])):
            curr_mask = transition_dict['states'][i]
            next_mask = transition_dict['next_states'][i]
            reward = transition_dict['rewards'][i].to(self.device)
            
            # 处理 done (如果是 tensor 就取值，如果是 bool/int 直接用)
            done = transition_dict['dones'][i]
            if isinstance(done, torch.Tensor):
                done = done.to(self.device)
            
            with torch.no_grad():
                # Critic 输入: (graph_batch, mask) -> 输出 [batch_size, 1]
                next_value = self.critic(graph_batch, next_mask)
                curr_value = self.critic(graph_batch, curr_mask)
                
                # 计算 TD Target: r + gamma * V_next * (1-done)
                target = reward + self.gamma * next_value * (1 - done)
                delta = target - curr_value
                
                td_target_list.append(target)
                td_delta_list.append(delta)

        # 拼接所有的 TD error 和 Target
        td_targets = torch.cat(td_target_list).view(-1, 1)
        td_deltas = torch.cat(td_delta_list).view(-1, 1)
        
        # 计算优势函数 Advantage
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_deltas.cpu()).to(self.device)
        
        # 2. 计算旧策略的 Log Probs (Old Log Probs)
        old_log_probs_list = []
        for i in range(len(transition_dict['states'])):
            curr_mask = transition_dict['states'][i]
            # 重新跑一遍 actor 获取当前 mask 下的所有边概率
            probs, _ = self.actor(graph_batch, curr_mask)
            
            # 取出当初实际执行的那个动作的概率
            # transition_dict['actions'][i] 是当前 step 选中的边索引
            action_indices = transition_dict['actions'][i]
            
            selected_probs = probs[action_indices]
            old_log_probs_list.append(torch.log(selected_probs + 1e-8))
            
        old_log_probs = torch.cat(old_log_probs_list).view(-1, 1).detach()

        # 3. PPO 更新循环
        for _ in range(self.epochs):
            new_log_probs_list = []
            entropy_list = []
            curr_values_list = []
            
            # 这一步比较慢，但逻辑正确：重新计算每个 step 的概率和价值
            for i in range(len(transition_dict['states'])):
                curr_mask = transition_dict['states'][i]
                action_indices = transition_dict['actions'][i]
                
                # Actor forward
                probs, _ = self.actor(graph_batch, curr_mask)
                selected_probs = probs[action_indices]
                new_log_probs_list.append(torch.log(selected_probs + 1e-8))
                
                # Entropy
                entropy_list.append(-(probs * torch.log(probs + 1e-8)).sum())
                
                # Critic forward
                curr_values_list.append(self.critic(graph_batch, curr_mask))

            new_log_probs = torch.cat(new_log_probs_list).view(-1, 1)
            curr_values = torch.cat(curr_values_list).view(-1, 1)
            entropy = torch.stack(entropy_list).mean() # Entropy 取平均
            
            # PPO Loss 计算
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            
            actor_loss = torch.mean(-torch.min(surr1, surr2)) - 0.01 * entropy
            critic_loss = torch.mean(F.mse_loss(curr_values, td_targets))
            
            # 反向传播
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

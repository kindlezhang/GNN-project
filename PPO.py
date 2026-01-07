import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import Sequential, Linear, ReLU, ModuleList, Softmax, ELU, Sigmoid
from module.utils import rl_utils  
import torch
import torch.nn.functional as F
import gymnasium as gym
from module.utils.reorganizer import relabel_graph, filter_correct_data, filter_correct_data_batch
from torch_geometric.utils import softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
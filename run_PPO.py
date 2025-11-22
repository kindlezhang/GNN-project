from module.gnn_model_zoo.mutag_gnn import MutagNet
from module.data_loader_zoo.mutag_dataloader import Mutagenicity

from torch_geometric.data import DataLoader

from module.utils import *
from module.utils.reorganizer import relabel_graph, filter_correct_data, filter_correct_data_batch
from module.utils.parser import parse_args
# from module.utils.logging import Logger

from rc_explainer_pool import RC_Explainer, RC_Explainer_pro, RC_Explainer_Batch, RC_Explainer_Batch_star
from train_test_pool_batch3 import train_policy
from train_test_pool_graph import explain_graphs_with_mcts
from PPO_copy import *
from tqdm import tqdm 

# 优先使用 MPS (Mac), 其次 CUDA, 最后 CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def configuration(dataset_name):
    '''Return dataset-specific configurations.'''
    configs = dict()
    if dataset_name in ['vg']:
        configs['_hidden_size'] = 128
        configs['_num_labels'] = 5
        configs['debias_flag'] = False
        configs['topN'] = None
        configs['batch_size'] = 64
        configs['scope'] = 'part'

        configs['train_dataset'] = Visual_Genome('Data/VG', mode='training')
        configs['test_dataset'] = Visual_Genome('Data/VG', mode='testing')

        configs['topK_ratio'] = 0.1

    elif dataset_name in ['ba3']:
        configs['_hidden_size'] = 64
        configs['_num_labels'] = 3
        configs['debias_flag'] = True
        configs['topN'] = 5
        configs['batch_size'] = 64
        configs['scope'] = 'part'

        configs['train_dataset'] = BA3Motif('Data/BA3', mode='training')
        configs['test_dataset'] = BA3Motif('Data/BA3', mode='testing')

        configs['topK_ratio'] = 10

    elif dataset_name in ['mutag']:
        configs['_hidden_size'] = 32
        configs['_num_labels'] = 2
        configs['debias_flag'] = False
        configs['topN'] = None
        configs['batch_size'] = 32
        configs['scope'] = 'all'

        configs['train_dataset'] = Mutagenicity('Data/MUTAG', mode='training')
        configs['test_dataset'] = Mutagenicity('Data/MUTAG', mode='testing')

        configs['topK_ratio'] = 0.1

    elif dataset_name in ['reddit5k']:
        configs['_hidden_size'] = 32
        configs['_num_labels'] = 5
        configs['debias_flag'] = False
        configs['topN'] = None
        configs['batch_size'] = 64
        configs['scope'] = 'part'

        configs['train_dataset'] = Reddit5k('Data/Reddit5k', mode='training')
        configs['test_dataset'] = Reddit5k('Data/Reddit5k', mode='testing')

        configs['topK_ratio'] = 0.1

    return configs

if __name__ == '__main__':

    set_seed(19930819)
    args = parse_args()
    args.dataset_name = "mutag"

    # if not torch.cuda.is_available():
    #     args.dataset_name = 'reddit5k'
    #     args.lr = 0.0001
    #     args.l2 = 0.0001
    #     args.reward_mode = 'mutual_info'

    dataset_name = args.dataset_name

    # get the configuration for a specific dataset
    configs = configuration(dataset_name)

    _hidden_size = configs['_hidden_size']
    _num_labels = configs['_num_labels']
    debias_flag = configs['debias_flag']
    topN = configs['topN']
    batch_size = configs['batch_size']
    scope = configs['scope']
    topK_ratio =  configs['topK_ratio'] 

    # get the trianing & testing datasets
    train_dataset = configs['train_dataset']
    test_dataset = configs['test_dataset']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    path = 'params/%s_net.pt' % dataset_name
    # if torch.cuda.is_available():
    #     model = torch.load(path, map_location=lambda storage, loc: storage.cuda(0))
    # elif torch.backends.mps.is_available():
    #     model = torch.load(path, map_location=torch.device('mps'))
    # else:
    #     model = torch.load(path, map_location=torch.device('cpu'))
    # model.eval()

    # if torch.cuda.is_available():
    #     model_1 = torch.load(path, map_location=lambda storage, loc: storage.cuda(0))
    # elif torch.backends.mps.is_available():
    #     model_1 = torch.load(path, map_location=torch.device('mps'))
    # else:
    #     model_1 = torch.load(path, map_location=torch.device('cpu'))
    # model_1.eval()

    model = MutagNet(2)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path, map_location='cuda:0'))   
    elif torch.backends.mps.is_available():
        model.load_state_dict(torch.load(path, map_location='mps')) 
    else:
        model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()

    model_1 = MutagNet(2)
    if torch.cuda.is_available():
        model_1.load_state_dict(torch.load(path, map_location='cuda:0'))   
    elif torch.backends.mps.is_available():
        model_1.load_state_dict(torch.load(path, map_location='mps')) 
    else:
        model_1.load_state_dict(torch.load(path, map_location='cpu'))
    model_1.eval()


    # refine the datasets and data loaders
    train_dataset, train_loader = filter_correct_data_batch(model, train_dataset, train_loader, 'training',
                                                            batch_size=batch_size)
    test_dataset, test_loader = filter_correct_data_batch(model, test_dataset, test_loader, 'testing',
                                                          batch_size=1)
    
    # ================================================================================

    actor_lr = 1e-3  # 策略网络学习率
    critic_lr = 1e-2  # 价值网络学习率
    num_episodes = 2  # 训练的总回合数
    hidden_dim = _hidden_size  # 隐藏层维度
    gamma = 0.98  # 折扣因子
    lmbda = 0.95  # GAE 参数
    epochs = 10  # 每次更新的轮数
    eps = 0.2  # PPO 截断范围
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")  # 设备设置（优先使用 GPU）
    gnn_model = model_1
    num_labels = _num_labels
    topK_ratio = topK_ratio 

    agent = PPO_Graph(gnn_model, num_labels, hidden_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device)

    return_list = []

    pbar = tqdm(range(num_episodes), desc="PPO Training", unit="episode")

    # 5. 自定义训练循环 (替代 rl_utils.train_on_policy_agent)
    for i_episode in pbar:
        episode_return = 0
        
        batch_iter = tqdm(train_loader, desc=f"Episode {i_episode} Batches", leave=False, unit="batch")

        # 遍历每一个 Batch (相当于 Gym 里的一个 Episode)
        for graph_batch in batch_iter:
            graph_batch = graph_batch.to(device)
            batch_size = graph_batch.num_graphs
            
            # === 初始化状态 ===
            # 初始状态：所有边都没选中 (全是 False)
            # num_edges 是这个 batch 里所有图的边总数
            current_state_mask = torch.zeros(graph_batch.num_edges, dtype=torch.bool).to(device)
            
            # 存储轨迹数据的字典
            transition_dict = {
                'states': [], 
                'actions': [], 
                'next_states': [], 
                'rewards': [], 
                'dones': []
            }
            
            # === 解释步骤 (Step Loop) ===
            # 假设我们每张图最多选 ratio * graph.edge_num 条关键边，或者直到 Done
            max_steps_per_graph = max(1, int(graph_batch.num_edges * topK_ratio / batch_size))
            print(max_steps_per_graph)
            
            for t in range(max_steps_per_graph):
                # 1. 智能体动作: 决定选哪条边
                # 返回的 action 是全局 edge_index 里的索引
                action_idx = agent.take_action(graph_batch, current_state_mask)
                
                # 2. 执行动作 (Step)
                next_state_mask = current_state_mask.clone()
                # 将选中的边设为 True
                # 注意：这里需要处理 batch 维度，agent.take_action 应该返回 batch 里每个图选的边
                next_state_mask[action_idx] = True
                
                # 3. 计算奖励
                reward = calculate_reward(model_1, graph_batch, next_state_mask)
                
                # 4. 判断结束 (Done)
                # 这里简单设定：到了 max_steps 就结束
                done = False if t < max_steps_per_graph - 1 else True
                
                # 5. 存储 Transition
                # 注意：State 是 Mask，Reward 是 Tensor
                transition_dict['states'].append(current_state_mask)
                transition_dict['actions'].append(action_idx)
                transition_dict['next_states'].append(next_state_mask)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                
                # 更新状态
                current_state_mask = next_state_mask
                episode_return += reward.mean().item() # 记录平均奖励用于画图
                
            # === PPO 更新 ===
            # 一个 Batch 的数据跑完后，进行一次 PPO 更新
            # 注意：我们需要把 list 转换一下，并且 update 函数需要接收 graph_batch
            
            # 这里的处理需要配合 PPO.update 的具体实现
            # 简单做法：把 transition_dict 里的 list stack 起来
            # 这里的 update 需要传入 graph_batch，因为 ValueNet 需要它
            agent.update(transition_dict, graph_batch)

        # 记录这一轮的平均回报
        return_list.append(episode_return)
        print(f"Episode {i_episode}, Return: {episode_return:.4f}")

    # 6. 绘图
    plt.plot(list(range(len(return_list))), return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO Graph Explainer')
    plt.show()
    

import networkx as nx
from torch_geometric.utils import to_networkx

def visualize_interpretation(agent, loader, device, num_samples=5):
    """
    从 loader 中取几个图，运行 Agent，画出原图和被选中的子图
    """
    agent.actor.eval() # 切换到评估模式
    
    # 从 loader 中获取数据
    iterator = iter(loader)
    
    for i in range(num_samples):
        try:
            data = next(iterator)
        except StopIteration:
            break
            
        data = data.to(device)
        
        # === 1. 运行 Agent 进行选边 (与训练逻辑一致) ===
        num_edges = data.num_edges
        current_mask = torch.zeros(num_edges, dtype=torch.bool).to(device)
        
        # 设定步数 (MUTAG 一般选 5-10 条边)
        max_steps = 10 
        
        print(f"\nSample {i+1}: Graph Label: {data.y.item()}")
        
        for t in range(max_steps):
            # 预测
            with torch.no_grad():
                # 注意：take_action 内部可能有 sample，评估时你可能想要 deterministic (argmax)
                # 但为了看 PPO 的行为，保持 sample 也可以，或者修改 take_action 支持 deterministic
                action_idx = agent.take_action(data, current_mask)
            
            # 更新 Mask
            current_mask[action_idx] = True
            
        # === 2. 打印选中的边 ===
        selected_indices = torch.nonzero(current_mask).squeeze().cpu().numpy()
        print(f"Selected Edge Indices: {selected_indices}")
        
        # === 3. 画图 ===
        # 转换为 NetworkX 对象
        G = to_networkx(data, to_undirected=True)
        
        # 设置布局
        pos = nx.kamada_kawai_layout(G)
        
        plt.figure(figsize=(10, 5))
        
        # --- 左图：原始图 ---
        plt.subplot(1, 2, 1)
        plt.title("Original Graph")
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=300)
        
        # --- 右图：解释子图 (高亮选中的边) ---
        plt.subplot(1, 2, 2)
        plt.title("Explained Subgraph (Red Edges)")
        
        # 先画所有节点和淡色的底边
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300)
        nx.draw_networkx_labels(G, pos)
        nx.draw_networkx_edges(G, pos, edge_color='lightgray', alpha=0.5)
        
        # 再画选中的边 (红色，加粗)
        # 注意：PyG 的边是双向的，NetworkX如果是无向图，需要处理一下索引映射
        # 这里简化处理：将 PyG 选中的边转换为 (u, v) 元组列表
        edge_index = data.edge_index.cpu().numpy()
        selected_edges = []
        
        # 获取被选中边的 (u, v)
        if selected_indices.ndim == 0: # 只选了一条边的情况
             u, v = edge_index[0, selected_indices], edge_index[1, selected_indices]
             selected_edges.append((u, v))
        else:
            for idx in selected_indices:
                u, v = edge_index[0, idx], edge_index[1, idx]
                selected_edges.append((u, v))
        
        # 在 NetworkX 图中高亮这些边
        nx.draw_networkx_edges(G, pos, edgelist=selected_edges, edge_color='red', width=2.0)
        
        plt.show()


        # ... (之前的训练代码) ...
    
    # 6. 绘图 (Loss/Return 曲线)
    plt.plot(list(range(len(return_list))), return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('PPO Graph Explainer')
    plt.show()
    
    # === 新增：可视化结果 ===
    print("开始可视化解释结果...")
    # 使用 test_loader 查看测试集的效果
    visualize_interpretation(agent, test_loader, device, num_samples=3)



def evaluate_fidelity(agent, loader, gnn_model, device):
    agent.actor.eval()
    gnn_model.eval()
    
    fidelity_scores = []
    
    for data in loader:
        data = data.to(device)
        num_edges = data.num_edges
        current_mask = torch.zeros(num_edges, dtype=torch.bool).to(device)
        
        # 1. 原图预测概率
        with torch.no_grad():
            full_logits = gnn_model(data.x, data.edge_index, data.edge_attr, data.batch)
            full_pred = full_logits.argmax(dim=1)
            full_probs = F.softmax(full_logits, dim=1)
            original_prob = full_probs[0, full_pred].item()
        
        # 2. 选边
        for _ in range(10): # 假设选10步
            with torch.no_grad():
                 action = agent.take_action(data, current_mask)
                 current_mask[action] = True
        
        # 3. 子图预测概率
        subgraph = relabel_graph(data, current_mask)
        with torch.no_grad():
            sub_logits = gnn_model(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)
            sub_probs = F.softmax(sub_logits, dim=1)
            sub_prob = sub_probs[0, full_pred].item() # 看原预测类别的概率变化
            
        # Fidelity+ = 原图概率 - 子图概率 (越小越好，说明子图保留了原图的信息)
        # 或者简单的：Accuracy Drop
        fidelity_scores.append(original_prob - sub_prob)
        
    print(f"Average Fidelity Impact: {np.mean(fidelity_scores):.4f}")
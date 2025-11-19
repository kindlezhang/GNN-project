import torch
import math
import torch.nn.functional as F
from module.utils.reorganizer import relabel_graph
from torch.distributions import Bernoulli, Categorical
from module.utils import *
from module.utils.reorganizer import relabel_graph, filter_correct_data
from tqdm import tqdm
from torch_scatter import scatter_max
import numpy as np
from tqdm import tqdm
from copy import deepcopy

EPS = 1
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------
# MCTS Node (UCT style, no prior P)
# -----------------------------

class MCTSNode:
    def __init__(self, state, parent=None, edge_index=None):
        # state: torch.BoolTensor mask of shape [num_edges]
        self.state = state.clone().to(device)
        self.parent = parent
        self.children = {}        # action_idx -> child_node
        self.N = 0                # visit count
        self.W = 0.0              # total value
        self.Q = 0.0              # mean value = W/N
        self.edge_index = edge_index  # 存储图的边信息

    def is_terminal(self, max_budget):
        return int(self.state.sum().item()) >= max_budget
    
    def get_available_actions(self):
        """
        当前节点的可选择动作（未被选择的边）
        """
        return (~self.state).nonzero(as_tuple=False).flatten().tolist()

    def get_neighbor_actions(self):
        """
        返回当前已选边的邻接边（排除已选边），如果没有邻接边 fallback 全图未选边
        """
        available_actions = self.get_available_actions()

        selected_edges = self.state.nonzero(as_tuple=False).flatten().tolist()
        if len(selected_edges) == 0:
            # 根节点没有已选边，返回所有未选边
            return available_actions

        if self.edge_index is None:
            # 没有边信息，返回所有未选边
            return available_actions

        neighbor_edges = set()
        for e in selected_edges:
            nodes = self.edge_index[:, e].tolist()  # 当前边的两个节点
            for n in nodes:
                # 找所有与节点 n 相连的边
                connected_edges = (self.edge_index[0] == n).nonzero(as_tuple=False).flatten().tolist()
                connected_edges += (self.edge_index[1] == n).nonzero(as_tuple=False).flatten().tolist()
                neighbor_edges.update(connected_edges)

        # 去掉已选过的边
        neighbor_edges = list(neighbor_edges - set(selected_edges))
        # 只保留还未被选择的边
        neighbor_edges = [e for e in neighbor_edges if e in available_actions]

        if len(neighbor_edges) == 0:
            # fallback 到全图未选边
            neighbor_edges = available_actions
        
        return neighbor_edges

    
# -----------------------------
# Rollout (single-graph) - random playout to budget
# -----------------------------

# def rollout_random(node_state, graph, model, max_budget):
#     """
#     node_state: torch.BoolTensor [num_edges] mask (single graph)
#     graph_single: single graph object
#     use_model_greedy: optional heuristic
#     """
#     state = node_state.clone()
#     steps = 0
#     chosen_edges = state.nonzero(as_tuple=False).flatten().tolist()  # 存储本次 rollout 选择的边

#     while int(state.sum().item()) < max_budget:
#         avail = (~state).nonzero(as_tuple=False).flatten()
#         if len(avail) == 0:
#             break

#         # 随机选择一个边
#         a = int(random.choice(avail.tolist()))
#         state[a] = True
#         chosen_edges.append(a)

#         steps += 1

#     # compute reward
#     model.eval()
#     subgraph = relabel_graph(graph, state)
#     sub_pred = F.softmax(model(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch).detach())
#     full_pred = F.softmax(model(graph.x, graph.edge_index,
#                                 graph.edge_attr, graph.batch).detach())
#     reward_vec = get_reward(full_pred, sub_pred, graph.y, mode='mutual_info')
#     reward = float(reward_vec.mean().item()) # 以免是多图

#     # --- 打印 rollout 信息 ---
#     print(f"Rollout chosen edges: {chosen_edges}, reward: {reward:.4f}")

#     return reward

def rollout_random(model, node_state, graph, max_budget, use_neighbor_expand=True):
    """
    node_state: torch.BoolTensor [num_edges] mask (single graph)
    graph: single graph object
    use_neighbor_expand: bool, 是否优先选择邻接边
    """
    state = node_state.clone()
    chosen_edges = state.nonzero(as_tuple=False).flatten().tolist()

    while int(state.sum().item()) < max_budget:
        # 可选择边
        unselected = (~state).nonzero(as_tuple=False).flatten().tolist()
        if len(unselected) == 0:
            break

        # ----------- 邻居扩展逻辑 -----------
        if use_neighbor_expand and len(chosen_edges) > 0:

            neighbor_edges = set()

            # 对每个已选边，找与其相邻的边
            for e in chosen_edges:
                # edge_index: [2, num_edges]
                u, v = graph.edge_index[:, e].tolist()

                # 找全部连接 u 或 v 的边
                connected_u = (graph.edge_index[0] == u).nonzero(as_tuple=False).flatten().tolist()
                connected_u += (graph.edge_index[1] == u).nonzero(as_tuple=False).flatten().tolist()

                connected_v = (graph.edge_index[0] == v).nonzero(as_tuple=False).flatten().tolist()
                connected_v += (graph.edge_index[1] == v).nonzero(as_tuple=False).flatten().tolist()

                neighbor_edges.update(connected_u)
                neighbor_edges.update(connected_v)

            # 移除已选边 
            neighbor_edges = neighbor_edges - set(chosen_edges)

            # 仅保留未选边 
            # neighbor_edges = neighbor_edges & set(unselected)

            # 如果有邻居边，则只从邻居选择
            if len(neighbor_edges) > 0:
                avail = sorted(neighbor_edges)      # 排序可选，不排序也行
            else:
                avail = unselected                  # fallback：全部未选边
        else:
            avail = unselected

        # ----------- 选择动作 -----------
        a = random.choice(avail)
        state[a] = True
        chosen_edges.append(a)


    # 计算 reward
    model.eval()
    subgraph = relabel_graph(graph, state)
    sub_pred = F.softmax(model(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch).detach(),dim=-1)
    full_pred = F.softmax(model(graph.x, graph.edge_index, graph.edge_attr, graph.batch).detach(),dim=-1)
    reward_vec = get_reward(full_pred, sub_pred, graph.y, mode='mutual_info')
    reward = float(reward_vec.mean().item())

    print(f"Rollout chosen edges: {chosen_edges}, reward: {reward:.4f}")
    return reward


def get_neighbor_edges(state, graph):
    """
    返回当前已选边的邻接边索引（未选的）
    state: [num_edges] bool tensor, 已选边
    graph.edge_index: [2, num_edges] tensor
    """
    num_edges = state.size(0)
    selected_edges = state.nonzero(as_tuple=False).flatten()
    
    if len(selected_edges) == 0:
        # 如果还没选任何边，随机选择任意未选边
        return (~state).nonzero(as_tuple=False).flatten()

    # 找出已选边涉及的节点
    selected_nodes = torch.unique(graph.edge_index[:, selected_edges])
    
    # 所有未选边
    avail_edges = (~state).nonzero(as_tuple=False).flatten()
    if len(avail_edges) == 0:
        return torch.tensor([], device=state.device, dtype=torch.long)

    # 只保留至少有一个端点在 selected_nodes 的边
    edge_nodes = graph.edge_index[:, avail_edges]  # [2, num_avail]
    mask = (edge_nodes[0].unsqueeze(0) == selected_nodes.unsqueeze(1)).any(0) | \
           (edge_nodes[1].unsqueeze(0) == selected_nodes.unsqueeze(1)).any(0)

    neighbor_edges = avail_edges[mask]
    if len(neighbor_edges) == 0:
        # 如果没有邻接边可选，则退化为任意未选边
        neighbor_edges = avail_edges

    return neighbor_edges


# -----------------------------
# MCTS search (single graph)
# -----------------------------

# def mcts_search_single(root_state, graph_single, model, max_budget,
#                        num_simulations=200, c_uct=1.0, rollout_limit=None, use_model_greedy=False):
#     """
#     graph_single: a single-graph object (batch size == 1)
#     root_state: torch.BoolTensor length num_edges for this graph
#     """
#     root = MCTSNode(root_state, parent=None)

#     for sim in range(num_simulations):
#         node = root
#         path = [node]

#         # Selection: go down using UCT until an expandable node
#         while node.is_fully_expanded() and not node.is_terminal(max_budget):
#             best_score = -1e9
#             best_child = None
#             for a, child in node.children.items():
#                 # UCT
#                 parent_N = node.N
#                 score = child.Q + c_uct * math.sqrt(math.log(parent_N + 1.0) / (child.N + 1.0))
#                 if score > best_score:
#                     best_score = score
#                     best_child = child
#             if best_child is None:
#                 break
#             node = best_child
#             path.append(node)

#         # If terminal, value is immediate (we could also compute exact reward for terminal node)
#         if node.is_terminal(max_budget):
#             # evaluate terminal reward
#             v = rollout_random(node.state, graph_single, model, max_budget, rollout_limit, use_model_greedy)
#         else:
#             # Expansion: pick one untried action (randomly) and expand
#             if len(node.untried_actions) > 0:
#                 a = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
#                 child_state = node.state.clone()
#                 child_state[a] = True
#                 child = MCTSNode(child_state, parent=node)
#                 node.children[a] = child
#                 node = child
#                 path.append(node)

#             # Simulation / rollout from the new node
#             v = rollout_random(node.state, graph_single, model, max_budget, rollout_limit, use_model_greedy)

#         # Backpropagate
#         for n in path:
#             n.N += 1
#             n.W += v
#             n.Q = n.W / (n.N + 0.0)

#     # After simulations: choose action with highest visit count (or highest Q)
#     if len(root.children) == 0:
#         # fallback: choose a random available action
#         avail = (~root.state).nonzero(as_tuple=False).flatten()
#         if len(avail) == 0:
#             return None, None
#         action = int(random.choice(avail.tolist()))
#         visit_dist = None
#     else:
#         visit_counts = {a: child.N for a, child in root.children.items()}
#         total = sum(visit_counts.values()) + EPS
#         visit_dist = {a: cnt / total for a, cnt in visit_counts.items()}
#         # pick action with max visits
#         action = max(visit_counts.items(), key=lambda x: x[1])[0]

#     return int(action), visit_dist

# # -----------------------------
# # Batch wrapper: run single MCTS per graph in batch
# # -----------------------------
# def mcts_search_batch(graph, model, max_budget_per_graph, num_simulations=200, c_uct=1.0, rollout_limit=None, use_model_greedy=False):
    """
    graph: batched graph where graph.batch indicates graph indices
    max_budget_per_graph: dict or int. If int, use same budget for all; if dict/list use per-graph budgets.
    returns: actions_per_graph list, visit_dists list
    """
    batch_size = graph.y.size(0)
    actions = [None] * batch_size
    dists = [None] * batch_size

    # split graph into per-graph objects: easiest is to create subgraphs via masks
    for g_idx in range(batch_size):
        mask = graph.batch == g_idx
        # build a subgraph object with same attrs expected by relabel_graph/model
        sub = deepcopy(graph)
        # keep only indices for edges/nodes belonging to g_idx:
        # If your graph uses edge_index referencing nodes, make sure relabel_graph supports batched input
        # Here we simply create a mask of edges belonging to g_idx and transform into a "single-graph" object:
        edges_mask = mask.clone()
        single_graph = deepcopy(graph)
        # set attributes expected by model/relabel_graph properly:
        single_graph.x = graph.x[graph.batch == g_idx]
        single_graph.edge_index = graph.edge_index[:, edges_mask]
        single_graph.edge_attr = graph.edge_attr[edges_mask] if hasattr(graph, 'edge_attr') else None
        single_graph.batch = torch.zeros(single_graph.x.size(0), dtype=torch.long, device=device)
        single_graph.y = graph.y[g_idx].unsqueeze(0)
        single_graph.num_edges = int(edges_mask.sum().item())
        # root state for single graph: length = num_edges (we assume edges indexed consecutively per-graph — adjust if needed)
        root_state = torch.zeros(single_graph.num_edges, dtype=torch.bool, device=device)

        mb = max_budget_per_graph if isinstance(max_budget_per_graph, int) else max_budget_per_graph[g_idx]
        a, dist = mcts_search_single(root_state, single_graph, model, mb,
                                     num_simulations=num_simulations, c_uct=c_uct,
                                     rollout_limit=rollout_limit, use_model_greedy=use_model_greedy)
        actions[g_idx] = a
        dists[g_idx] = dist

    return actions, dists


# def explain_graphs_with_mcts_single(graph, model, max_budget, num_simulations=200, c_uct=1.0, rollout_limit=None, use_neighbor_expand=True):
#     root_state = torch.zeros(graph.num_edges, dtype=torch.bool, device=device)
#     root = MCTSNode(root_state, parent=None, edge_index=graph.edge_index)

#     for sim in range(num_simulations):
#         node = root
#         path = [node]

#         # -------- Root special handling --------
#         if node.N == 0:
#             # 第一次 simulation 扩展 root 所有 untried actions

#             if use_neighbor_expand:
#                 actions_to_expand = node.get_neighbor_actions()
#             else:
#                 actions_to_expand = node.untried_actions[:]

#             for a in actions_to_expand:
#                 child_state = node.state.clone()
#                 child_state[a] = True
#                 child = MCTSNode(child_state, parent=node)
#                 node.children[a] = child
#                 node.untried_actions.remove(a)
#             # 从 root children 随机选择一个 child 进行 rollout
#             a = random.choice(list(node.children.keys()))
#             node = node.children[a]
#             path.append(node)
#             v = rollout_random(node.state, graph, model, max_budget=max_budget)
#         else:
#             # -------- Selection down the tree --------
#             while node.children and not node.is_terminal(max_budget):
#                 # UCT 选择最大 child
#                 best_score = -1e9
#                 best_child = None
#                 for a, child in node.children.items():
#                     score = child.Q + c_uct * math.sqrt(math.log(node.N) / (child.N + EPS))
#                     if score > best_score:
#                         best_score = score
#                         best_child = child
#                 if best_child is None:
#                     break
#                 node = best_child
#                 path.append(node)

#             # -------- Leaf node handling --------
#             if not node.is_terminal(max_budget) and not node.children:
#                 if node.N == 0:
#                     # 新 leaf → rollout
#                     v = rollout_random(node.state, graph, model, max_budget=max_budget)
#                 else:
#                     # 已访问 leaf → 扩展所有 untried actions，再随机 rollout
#                     for a in node.untried_actions[:]:
#                         child_state = node.state.clone()
#                         child_state[a] = True
#                         child = MCTSNode(child_state, parent=node)
#                         node.children[a] = child
#                         node.untried_actions.remove(a)
#                     a = random.choice(list(node.children.keys()))
#                     node = node.children[a]
#                     path.append(node)
#                     v = rollout_random(node.state, graph, model, max_budget=max_budget)
#             else:
#                 # terminal 节点
#                 # 改一下
#                 v = rollout_random(node.state, graph, model, max_budget=max_budget)

#         # -------- Backpropagation --------
#         for n in path:
#             n.N += 1
#             n.W += v
#             n.Q = n.W / n.N
        
#         # -------- Print this simulation's chosen actions --------
#         sim_actions = []
#         for n in path[1:]:  # 跳过 root 节点
#             for a, child in n.parent.children.items():
#                 if child is n:
#                     sim_actions.append(a)
#                     break
#         print(f"Simulation {sim+1}: path actions = {sim_actions}")

#     # -------- Extract best subgraph --------
#     selected_edges = []
#     node = root
#     while not node.is_terminal(max_budget) and node.children:
#         a, node = max(node.children.items(), key=lambda x: x[1].Q)
#         selected_edges.append(a)

#     subgraph_mask = torch.zeros(graph.num_edges, dtype=torch.bool, device=device)
#     subgraph_mask[selected_edges] = True
#     subgraph = relabel_graph(graph, subgraph_mask)

#     return selected_edges, subgraph



def explain_graphs_with_mcts_single(graph, model, max_budget, num_simulations=200, 
                                     c_uct=1.0, rollout_limit=None, use_neighbor_expand=True):
    """
    graph: single graph object
    max_budget: int, 最大选边数
    use_neighbor_expand: bool, 是否扩展邻接边
    """
    root_state = torch.zeros(graph.num_edges, dtype=torch.bool, device=device)
    root = MCTSNode(root_state, parent=None, edge_index=graph.edge_index)

    for sim in range(num_simulations):
        node = root
        path = [node]

        # -------- Root special handling --------
        if node.N == 0:
            # 扩展 root 所有未选边
            all_edges = list(range(graph.num_edges))
            for a in all_edges:
                child_state = node.state.clone()
                child_state[a] = True
                child = MCTSNode(child_state, parent=node, edge_index=node.edge_index)
                node.children[a] = child

            # 随机选一个 child 做 rollout
            a = random.choice(list(node.children.keys()))
            node = node.children[a]
            path.append(node)
            v = rollout_random(model, node.state, graph, max_budget=max_budget, use_neighbor_expand=use_neighbor_expand)

        else:
            # -------- Selection down the tree --------
            while node.children and not node.is_terminal(max_budget):
                best_score = -1e9
                best_child = None
                pos = 0
                for a, child in node.children.items():
                    score = child.Q + c_uct * math.sqrt(math.log(node.N) / (child.N + EPS))
                    # print(a, score)
                    if score > best_score:
                        best_score = score
                        best_child = child
                        pos = a
                # print("best score:", best_score, "best child action:", pos)
                if best_child is None:
                    break

                node = best_child
                path.append(node)
            print(" Selected path actions:", node.N, node.Q, node.W)

            # -------- Leaf node handling --------
            if not node.is_terminal(max_budget) and not node.children:

                # 第一次访问 leaf：直接 rollout
                if node.N == 0:
                    v = rollout_random(
                        model, node.state, graph,
                        max_budget=max_budget,
                        use_neighbor_expand=use_neighbor_expand
                    )
                # 第二次访问 leaf：扩展邻居边（或 fallback）
                else:
                    # 扩展邻接边或者 fallback
                    if use_neighbor_expand:
                        actions_to_expand = node.get_neighbor_actions()
                    else:
                        actions_to_expand = list(range(graph.num_edges))

                    print(" Neighbor actions:", actions_to_expand)
                
                    # # 过滤掉已经选过的
                    # selected_edges = node.state.nonzero(as_tuple=False).flatten().tolist()
                    # actions_to_expand = [a for a in actions_to_expand if a not in selected_edges]

                    # 扩展
                    for a in actions_to_expand:
                        child_state = node.state.clone()
                        child_state[a] = True
                        child = MCTSNode(child_state, parent=node, edge_index=node.edge_index)
                        node.children[a] = child    

                    # 随机选择一个 child 做 rollout
                    a = random.choice(list(node.children.keys()))
                    print(" Expanding leaf, chosen action:", a)
                    node = node.children[a]
                    path.append(node)
                    v = rollout_random(model, node.state, graph, max_budget=max_budget, use_neighbor_expand=use_neighbor_expand)
            else:
                # terminal 节点
                v = rollout_random(model, node.state, graph, max_budget=max_budget, use_neighbor_expand=use_neighbor_expand)

        # -------- Backpropagation --------
        # print("backward")
        for n in path:
            n.N += 1
            n.W += v
            n.Q = n.W / n.N

        # 打印本次 simulation 路径动作
        sim_actions = []
        for n in path[1:]:
            for a, child in n.parent.children.items():
                if child is n:
                    sim_actions.append(a)
                    break
        print(f"Simulation {sim+1}: path actions = {sim_actions}")

    # -------- Extract best subgraph --------
    selected_edges = []
    node = root
    while not node.is_terminal(max_budget) and node.children:
        a, node = max(node.children.items(), key=lambda x: x[1].Q)
        selected_edges.append(a)

    subgraph_mask = torch.zeros(graph.num_edges, dtype=torch.bool, device=device)
    subgraph_mask[selected_edges] = True
    subgraph = relabel_graph(graph, subgraph_mask)

    return selected_edges, subgraph



def explain_graphs_with_mcts(train_loader, model, topK_ratio,batch_size,
                              num_simulations=200, c_uct=1.0, rollout_limit=None, use_neighbor_expand=True):
    """
    对 train_loader 中的每个 graph 单独运行 MCTS，返回每个图的解释子图边索引和子图
    """
    results = []

    for graph in tqdm(iter(train_loader), total=len(train_loader)):
        graph = graph.to(device)
        # 根据 topK_ratio 计算 max_budget
        if topK_ratio < 1:
            max_budget = max(int(topK_ratio * graph.num_edges), 1)
        else:
            max_budget = int(topK_ratio)

        # 单图 MCTS
        selected_edges, subgraph = explain_graphs_with_mcts_single(graph, model, max_budget,
                                                                    num_simulations=num_simulations,
                                                                    c_uct=c_uct,
                                                                    rollout_limit=rollout_limit,
                                                                    use_neighbor_expand=use_neighbor_expand)
        results.append({
            "graph": graph,
            "selected_edges": selected_edges,
            "subgraph": subgraph
        })

    return results

def get_reward(full_subgraph_pred, new_subgraph_pred, target_y, mode='mutual_info'):
    """
    full_subgraph_pred, new_subgraph_pred: [num_graphs, num_classes] probability tensors
    target_y: [num_graphs] true labels
    """
    EPS = 1e-15  # 防止 log(0)


    target_y = target_y.to(device)

    if mode in ['mutual_info']:
        # KL-like term，可能为负 → 平移 +1
        reward = torch.sum(full_subgraph_pred * torch.log(new_subgraph_pred + EPS), dim=1).to(device)
        reward += (target_y == new_subgraph_pred.argmax(dim=1).to(device)).float()
        reward = reward + 1.0  # 保证非负

    elif mode in ['binary']:
        reward = (target_y == new_subgraph_pred.argmax(dim=1).to(device)).float()
        # 本身就是 0/1，非负

    elif mode in ['cross_entropy']:
        reward = torch.log(new_subgraph_pred + EPS)[range(target_y.size(0)), target_y]
        reward = reward + 1.0  # 平移保证非负

    else:
        raise ValueError(f"Unknown reward mode: {mode}")

    return reward
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

EPS = 1e-15
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------
# MCTS Node (UCT style, no prior P)
# -----------------------------

class MCTSNode:
    def __init__(self, state, parent=None):
        # state: torch.BoolTensor mask of shape [num_edges]
        self.state = state.clone().to(device)
        self.parent = parent
        self.children = {}        # action_idx -> child_node
        self.N = 0                # visit count
        self.W = 0.0              # total value
        self.Q = 0.0              # mean value = W/N
        self.untried_actions = (~self.state).nonzero(as_tuple=False).flatten().tolist()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self, max_budget):
        return int(self.state.sum().item()) >= max_budget
    
# -----------------------------
# Rollout (single-graph) - random playout to budget
# -----------------------------

def rollout_random(node_state, graph, model, max_budget):
    """
    node_state: torch.BoolTensor [num_edges] mask (single graph)
    graph_single: single graph object
    use_model_greedy: optional heuristic
    """
    state = node_state.clone()
    steps = 0
    chosen_edges = state.nonzero(as_tuple=False).flatten().tolist()  # 存储本次 rollout 选择的边

    while int(state.sum().item()) < max_budget:
        avail = (~state).nonzero(as_tuple=False).flatten()
        if len(avail) == 0:
            break

        # 随机选择一个边
        a = int(random.choice(avail.tolist()))
        state[a] = True
        chosen_edges.append(a)

        steps += 1

    # compute reward
    model.eval()
    subgraph = relabel_graph(graph, state)
    sub_pred = F.softmax(model(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch).detach())
    full_pred = F.softmax(model(graph.x, graph.edge_index,
                                graph.edge_attr, graph.batch).detach())
    reward_vec = get_reward(full_pred, sub_pred, graph.y, mode='mutual_info')
    reward = float(reward_vec.mean().item()) # 以免是多图

    # --- 打印 rollout 信息 ---
    print(f"Rollout chosen edges: {chosen_edges}, reward: {reward:.4f}")

    return reward


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


def explain_graphs_with_mcts_single(graph, model, max_budget, num_simulations=200, c_uct=1.0, rollout_limit=None):
    root_state = torch.zeros(graph.num_edges, dtype=torch.bool, device=device)
    root = MCTSNode(root_state, parent=None)

    for sim in range(num_simulations):
        node = root
        path = [node]

        # -------- Root special handling --------
        if node.N == 0:
            # 第一次 simulation 扩展 root 所有 untried actions
            for a in node.untried_actions[:]:
                child_state = node.state.clone()
                child_state[a] = True
                child = MCTSNode(child_state, parent=node)
                node.children[a] = child
                node.untried_actions.remove(a)
            # 从 root children 随机选择一个 child 进行 rollout
            a = random.choice(list(node.children.keys()))
            node = node.children[a]
            path.append(node)
            v = rollout_random(node.state, graph, model, max_budget=max_budget)
        else:
            # -------- Selection down the tree --------
            while node.children and not node.is_terminal(max_budget):
                # UCT 选择最大 child
                best_score = -1e9
                best_child = None
                for a, child in node.children.items():
                    score = child.Q + c_uct * math.sqrt(math.log(node.N) / (child.N + EPS))
                    if score > best_score:
                        best_score = score
                        best_child = child
                if best_child is None:
                    break
                node = best_child
                path.append(node)

            # -------- Leaf node handling --------
            if not node.is_terminal(max_budget) and not node.children:
                if node.N == 0:
                    # 新 leaf → rollout
                    v = rollout_random(node.state, graph, model, max_budget=max_budget)
                else:
                    # 已访问 leaf → 扩展所有 untried actions，再随机 rollout
                    for a in node.untried_actions[:]:
                        child_state = node.state.clone()
                        child_state[a] = True
                        child = MCTSNode(child_state, parent=node)
                        node.children[a] = child
                        node.untried_actions.remove(a)
                    a = random.choice(list(node.children.keys()))
                    node = node.children[a]
                    path.append(node)
                    v = rollout_random(node.state, graph, model, max_budget=max_budget)
            else:
                # terminal 节点
                # 改一下
                v = rollout_random(node.state, graph, model, max_budget=max_budget)

        # -------- Backpropagation --------
        for n in path:
            n.N += 1
            n.W += v
            n.Q = n.W / n.N
        
        # -------- Print this simulation's chosen actions --------
        sim_actions = []
        for n in path[1:]:  # 跳过 root 节点
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
                              num_simulations=200, c_uct=1.0, rollout_limit=None):
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
                                                                    rollout_limit=rollout_limit)
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
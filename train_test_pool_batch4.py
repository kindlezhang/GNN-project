import torch
import torch.nn.functional as F
from module.utils.reorganizer import relabel_graph
from torch.distributions import Bernoulli, Categorical
from module.utils import *
from module.utils.reorganizer import relabel_graph, filter_correct_data

from tqdm import tqdm
from torch_scatter import scatter_max

EPS = 1e-15
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")


def test_policy_all_with_gnd(rc_explainer, model, test_loader, topN=None):
    rc_explainer.eval()
    model.eval()

    topK_ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    acc_count_list = np.zeros(len(topK_ratio_list))

    precision_topN_count = 0.
    recall_topN_count = 0.

    with torch.no_grad():
        for graph in iter(test_loader):
            graph = graph.to(device)
            max_budget = graph.num_edges
            state = torch.zeros(max_budget, dtype=torch.bool)

            check_budget_list = [max(int(_topK * max_budget), 1) for _topK in topK_ratio_list]
            valid_budget = max(int(0.9 * max_budget), 1)

            for budget in range(valid_budget):
                available_actions = state[~state].clone()

                _, _, make_action_id, _ = rc_explainer(graph=graph, state=state, train_flag=False)

                available_actions[make_action_id] = True
                state[~state] = available_actions.clone()

                if (budget + 1) in check_budget_list:
                    check_idx = check_budget_list.index(budget + 1)
                    subgraph = relabel_graph(graph, state)
                    subgraph_pred = model(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)

                    acc_count_list[check_idx] += sum(graph.y == subgraph_pred.argmax(dim=1))

                if topN is not None and budget == topN - 1:
                    precision_topN_count += torch.sum(state*graph.ground_truth_mask[0])/topN
                    recall_topN_count += torch.sum(state*graph.ground_truth_mask[0])/sum(graph.ground_truth_mask[0])

    acc_count_list[-1] = len(test_loader)
    acc_count_list = np.array(acc_count_list)/len(test_loader)

    precision_topN_count = precision_topN_count / len(test_loader)
    recall_topN_count = recall_topN_count / len(test_loader)

    if topN is not None:
        print('\nACC-AUC: %.4f, Precision@5: %.4f, Recall@5: %.4f' %
              (acc_count_list.mean(), precision_topN_count, recall_topN_count))
    else:
        print('\nACC-AUC: %.4f' % acc_count_list.mean())
    print(acc_count_list)

    return acc_count_list.mean(), acc_count_list, precision_topN_count, recall_topN_count


def bias_detector(model, graph, valid_budget):
    pred_bias_list = []

    for budget in range(valid_budget):
        num_repeat = 2

        i_pred_bias = 0.
        for i in range(num_repeat):
            bias_selection = torch.zeros(graph.num_edges, dtype=torch.bool)

            ava_action_batch = graph.batch[graph.edge_index[0]]
            ava_action_probs = torch.rand(ava_action_batch.size()).to(device)
            _, added_actions = scatter_max(ava_action_probs, ava_action_batch)

            bias_selection[added_actions] = True
            bias_subgraph = relabel_graph(graph, bias_selection)
            bias_subgraph_pred = model(bias_subgraph.x, bias_subgraph.edge_index,
                                       bias_subgraph.edge_attr, bias_subgraph.batch).detach()

            i_pred_bias += bias_subgraph_pred / num_repeat

        pred_bias_list.append(i_pred_bias)

    return pred_bias_list

def get_reward(full_subgraph_pred, new_subgraph_pred, target_y, pre_reward, mode='mutual_info'):
    if mode in ['mutual_info']:
        reward = torch.sum(full_subgraph_pred * torch.log(new_subgraph_pred + EPS), dim=1)
        reward += 2 * (target_y == new_subgraph_pred.argmax(dim=1)).float() - 1.

    elif mode in ['binary']:
        target_y = target_y.to(device)
        new_subgraph_pred = new_subgraph_pred.to(device)
        reward = (target_y == new_subgraph_pred.argmax(dim=1)).float()
        reward = 2. * reward - 1.

    elif mode in ['cross_entropy']:
        reward = torch.log(new_subgraph_pred + EPS)[:, target_y]

    # reward += pre_reward
    reward += 0.97 * pre_reward

    return reward

# -----------------------------
# MCTS 节点类
# -----------------------------
class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state.clone()
        self.parent = parent
        self.children = {}
        self.N = 0  # 访问次数
        self.W = 0.  # 累积奖励
        self.Q = 0.  # 平均奖励
        self.untried_actions = (~state).nonzero(as_tuple=False).flatten()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self, max_budget):
        return self.state.sum() >= max_budget

    def expand(self):
        action = self.untried_actions[0]
        self.untried_actions = self.untried_actions[1:]
        new_state = self.state.clone()
        new_state[action] = True
        child = MCTSNode(new_state, parent=self)
        self.children[action.item()] = child
        return child, action.item()

    def best_ucb_child(self, policy_probs, c=1.0):
        best_score = -float('inf')
        best_action = None
        best_child = None
        for action, child in self.children.items():
            P = policy_probs[action]
            ucb = child.Q + c * P * ( (self.N+1)**0.5 / (1 + child.N) )
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child
        return best_child, best_action

# -----------------------------
# Rollout: 从节点随机或策略采样到终止
# -----------------------------
def rollout(node, graph, rc_explainer, model, max_budget, debias_flag=False, pred_bias_list=None):
    state = node.state.clone()
    while state.sum() < max_budget:
        avail_actions = (~state).nonzero(as_tuple=False).flatten()
        if len(avail_actions) == 0:
            break
        # 用策略网络生成概率
        _, action_probs, actions, _ = rc_explainer(graph, state, train_flag=False)
        action_probs = action_probs[avail_actions]
        action_probs = F.softmax(action_probs, dim=0)
        action_idx = torch.multinomial(action_probs, 1).item()
        action = avail_actions[action_idx]
        state[action] = True

    # 构建子图并计算 reward
    subgraph = relabel_graph(graph, state)
    subgraph_pred = model(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)
    if debias_flag:
        budget_idx = state.sum() - 1
        subgraph_pred = F.softmax(subgraph_pred - pred_bias_list[budget_idx]).detach()
    else:
        subgraph_pred = F.softmax(subgraph_pred).detach()
    full_pred = F.softmax(model(graph.x, graph.edge_index, graph.edge_attr, graph.batch).detach())
    reward = get_reward(full_pred, subgraph_pred, graph.y, pre_reward=torch.zeros(graph.y.size()).to(device))
    return reward.mean().item()  # 对 batch 求平均

# -----------------------------
# MCTS + PPO 训练
# -----------------------------
from tqdm import tqdm

def train_policy_mcts_ppo(rc_explainer, model, train_loader, test_loader, optimizer,
                          topK_ratio=0.1, debias_flag=False, topN=None,
                          num_episodes=30, num_simulations=20, c_ucb=1.0,
                          clip_ratio=0.2, gamma=0.97):

    best_acc_auc, best_acc_curve, best_pre, best_rec = 0, 0, 0, 0

    for ep in range(num_episodes):
        rc_explainer.train()
        model.eval()
        total_loss = 0.

        # 🔥 tqdm 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{num_episodes}", ncols=100)

        for graph in pbar:
            graph = graph.to(device)
            max_budget = max(int(topK_ratio * graph.num_edges), 1)
            current_state = torch.zeros(graph.num_edges, dtype=torch.bool).to(device)

            # 可选 debias
            if debias_flag:
                pred_bias_list = bias_detector(model, graph, max_budget)
            else:
                pred_bias_list = None

            log_probs = []
            rewards = []

            # -------------------------
            # 每个 budget 用 MCTS 搜索最优动作
            # -------------------------
            for budget_step in range(max_budget):

                root = MCTSNode(current_state)

                # 多次模拟
                for _ in range(num_simulations):
                    node = root
                    path = [node]

                    # Selection
                    while not node.is_terminal(max_budget) and node.is_fully_expanded():
                        _, policy_probs = rc_explainer(graph, node.state, train_flag=False)[:2]
                        policy_probs = F.softmax(policy_probs, dim=0)
                        node, _ = node.best_ucb_child(policy_probs, c=c_ucb)
                        path.append(node)

                    # Expansion
                    if not node.is_terminal(max_budget):
                        node, _ = node.expand()
                        path.append(node)

                    # Rollout
                    r = rollout(node, graph, rc_explainer, model, max_budget,
                                debias_flag, pred_bias_list)

                    # Backprop
                    for n in path:
                        n.N += 1
                        n.W += r
                        n.Q = n.W / n.N

                # 从 MCTS 选择动作
                best_child = max(root.children.values(), key=lambda n: n.Q)
                action = [k for k,v in root.children.items() if v==best_child][0]
                current_state[action] = True

                # PPO log prob
                _, action_probs, _, _ = rc_explainer(graph, current_state, train_flag=False)
                log_prob = torch.log(action_probs[action] + EPS)
                log_probs.append(log_prob)
                rewards.append(r)

            # -------------------------
            # 计算折扣奖励
            # -------------------------
            discounted_rewards = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                discounted_rewards.insert(0, R)

            discounted_rewards = torch.tensor(discounted_rewards).to(device)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + EPS)

            # PPO loss
            log_probs = torch.stack(log_probs)
            ratio = torch.exp(log_probs - log_probs.detach())
            clip_loss = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * discounted_rewards
            loss = -torch.mean(torch.min(ratio * discounted_rewards, clip_loss))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 更新 tqdm 显示的 loss
            pbar.set_postfix(loss=loss.item())

        # -------------------------
        # 测试
        # -------------------------
        ep_acc_auc, ep_acc_curve, ep_pre, ep_rec = test_policy_all_with_gnd(
            rc_explainer, model, test_loader, topN
        )

        print(f"[EP {ep+1}] Loss={total_loss:.4f}, ACC-AUC={ep_acc_auc:.4f}, "
              f"Pre@{topN}={ep_pre:.4f}, Rec@{topN}={ep_rec:.4f}")

        if ep_acc_auc >= best_acc_auc:
            best_acc_auc = ep_acc_auc
            best_acc_curve = ep_acc_curve
            best_pre = ep_pre
            best_rec = ep_rec
            rc_explainer.save_policy_net(path=None)

    return rc_explainer, best_acc_auc, best_acc_curve, best_pre, best_rec

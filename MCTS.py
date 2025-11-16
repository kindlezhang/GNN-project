好的，我帮你把你原来的 `train_policy` 函数改成 **MCTS + PPO/GRPO 训练版本**的伪代码示例，保持原来的 reward、debias、topN 逻辑不变，同时结合 PyTorch 的 PPO 更新思路。

下面是完整示意：

```python
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from module.utils.reorganizer import relabel_graph

EPS = 1e-15
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")

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
def train_policy_mcts_ppo(rc_explainer, model, train_loader, test_loader, optimizer,
                          topK_ratio=0.1, debias_flag=False, topN=None,
                          num_episodes=30, num_simulations=20, c_ucb=1.0, clip_ratio=0.2, gamma=0.97):

    best_acc_auc, best_acc_curve, best_pre, best_rec = 0, 0, 0, 0
    for ep in range(num_episodes):
        rc_explainer.train()
        model.eval()
        total_loss = 0.
        for graph in train_loader:
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

            # 每个 budget 用 MCTS 搜索最优动作
            for budget_step in range(max_budget):
                root = MCTSNode(current_state)
                # 多次模拟
                for _ in range(num_simulations):
                    node = root
                    path = [node]
                    # Selection & Expansion
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
                    r = rollout(node, graph, rc_explainer, model, max_budget, debias_flag, pred_bias_list)
                    # Backprop
                    for n in path:
                        n.N += 1
                        n.W += r
                        n.Q = n.W / n.N

                # 选择 MCTS 根节点最优动作
                best_child = max(root.children.values(), key=lambda n: n.Q)
                action = [k for k,v in root.children.items() if v==best_child][0]
                current_state[action] = True

                # 保存 log_prob + reward 用于 PPO
                _, action_probs, _, _ = rc_explainer(graph, current_state, train_flag=False)
                log_prob = torch.log(action_probs[action] + EPS)
                log_probs.append(log_prob)
                rewards.append(r)

            # 计算折扣奖励
            discounted_rewards = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                discounted_rewards.insert(0, R)
            discounted_rewards = torch.tensor(discounted_rewards).to(device)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + EPS)

            # PPO loss
            log_probs = torch.stack(log_probs)
            ratio = torch.exp(log_probs - log_probs.detach())  # old_log_prob假设用detach
            clip_loss = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * discounted_rewards
            loss = -torch.mean(torch.min(ratio*discounted_rewards, clip_loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 测试
        ep_acc_auc, ep_acc_curve, ep_pre, ep_rec = test_policy_all_with_gnd(rc_explainer, model, test_loader, topN)
        print(f"[EP {ep+1}] Loss={total_loss:.4f}, ACC-AUC={ep_acc_auc:.4f}, Pre@{topN}={ep_pre:.4f}, Rec@{topN}={ep_rec:.4f}")

        # 保存最优策略
        if ep_acc_auc >= best_acc_auc:
            best_acc_auc = ep_acc_auc
            best_acc_curve = ep_acc_curve
            best_pre = ep_pre
            best_rec = ep_rec
            rc_explainer.save_policy_net(path=None)

    return rc_explainer, best_acc_auc, best_acc_curve, best_pre, best_rec
```

---

### **特点说明**

1. **MCTS 代替原 beam search**：

   * 每步用 `num_simulations` 次模拟估计每个动作的 Q 值
   * 最终选择 Q 值最高的动作
2. **PPO 更新策略网络**：

   * 用 MCTS 的 reward 作为 advantage
   * 使用 clipped loss 保证更新稳定
3. **保持原有逻辑**：

   * `debias_flag`、`topN`、`reward_mode` 都可以复用
4. **可扩展**：

   * 可以改成 GRPO，只需要替换 PPO loss 为 GRPO 的图结构 loss

---

如果你需要，我可以帮你画一张**MCTS + PPO + GNN 子图选择的可视化流程图**，把 `state → MCTS → rollout → reward → PPO` 的数据流直观展示出来。

你希望我画吗？

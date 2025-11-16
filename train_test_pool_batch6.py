import torch
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
# MCTS Node (PUCT style)
# -----------------------------
class MCTSNode:
    def __init__(self, state, parent=None, prior=1.0):
        # state: torch.BoolTensor of shape [num_edges] (mask of selected edges)
        self.state = state.clone().to(device)
        self.parent = parent
        self.children = {}        # action_idx -> child_node
        self.P = prior            # prior probability for this node (from parent's policy for the action)
        self.N = 0                # visit count
        self.W = 0.0              # total value
        self.Q = 0.0              # mean value = W/N
        # list of actions (edge indices) that are not yet expanded
        self.untried_actions = (~self.state).nonzero(as_tuple=False).flatten().tolist()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal(self, max_budget):
        return int(self.state.sum().item()) >= max_budget

# -----------------------------
# Rollout using the policy network (PPO actor), batch support
# -----------------------------
def rollout_policy(node, graph, rc_explainer, model, max_budget, debias_flag=False, pred_bias_list=None, rollout_limit=None):
    """
    node.state: [total_edges] bool mask for the batch
    """
    state = node.state.clone()
    steps = 0
    batch_size = graph.y.size(0)
    while int(state.sum().item()) < max_budget:
        avail = (~state).nonzero(as_tuple=False).flatten()
        if len(avail) == 0:
            break

        with torch.no_grad():
            out = rc_explainer(graph, state, train_flag=False)
        policy_logits = out[1] if len(out) >= 2 else out[0]

        # construct probabilities per edge in batch
        probs_all = torch.zeros(graph.num_edges, device=device)
        for g_idx in range(batch_size):
            avail_g = avail[graph.batch[avail] == g_idx]
            if len(avail_g) == 0:
                continue
            if policy_logits.shape[0] == graph.num_edges:
                raw_g = policy_logits[avail_g]
            else:
                raw_g = policy_logits[g_idx][:len(avail_g)]  # assume outputs per graph
            raw_g = F.softmax(raw_g, dim=0)
            probs_all[avail_g] = raw_g

        # normalize
        if probs_all.sum() <= 0:
            probs_all[avail] = 1.0 / len(avail)
        else:
            probs_all = probs_all / (probs_all.sum() + EPS)

        action_idx = torch.multinomial(probs_all, 1).item()
        state[action_idx] = True
        steps += 1
        if rollout_limit is not None and steps >= rollout_limit:
            break

    # build subgraph and compute reward
    subgraph = relabel_graph(graph, state)
    subgraph_pred = model(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)
    if debias_flag:
        for g_idx in range(batch_size):
            budget_idx = int(state[graph.batch == g_idx].sum().item()) - 1
            subgraph_pred[graph.batch == g_idx] -= pred_bias_list[budget_idx][g_idx]
        subgraph_pred = F.softmax(subgraph_pred.detach())
    else:
        subgraph_pred = F.softmax(subgraph_pred.detach())

    full_pred = F.softmax(model(graph.x, graph.edge_index, graph.edge_attr, graph.batch).detach())
    reward_vec = get_reward(full_pred, subgraph_pred, graph.y, pre_reward=torch.zeros(graph.y.size(), device=device))
    return float(reward_vec.mean().item())


# -----------------------------
# MCTS search, batch support
# -----------------------------
def mcts_search(root_state, graph, rc_explainer, model, max_budget, num_simulations=20, c_puct=1.0, debias_flag=False, pred_bias_list=None, rollout_limit=None):
    batch_size = graph.y.size(0)
    root = MCTSNode(root_state, parent=None, prior=1.0)

    for sim in range(num_simulations):
        node = root
        path = [node]

        # Selection
        while node.is_fully_expanded() and not node.is_terminal(max_budget):
            best_score = -1e9
            best_child = None
            for a, child in node.children.items():
                score = child.Q + c_puct * child.P * ((node.N ** 0.5) / (1 + child.N))
                if score > best_score:
                    best_score = score
                    best_child = child
            if best_child is None:
                break
            node = best_child
            path.append(node)

        # Terminal check
        if node.is_terminal(max_budget):
            v = 0.0
        else:
            # Expansion
            if len(node.untried_actions) > 0:
                with torch.no_grad():
                    out = rc_explainer(graph, node.state, train_flag=False)
                policy_logits = out[1] if len(out) >= 2 else out[0]

                avail = (~node.state).nonzero(as_tuple=False).flatten()
                probs_all = torch.zeros(graph.num_edges, device=device)
                for g_idx in range(batch_size):
                    avail_g = avail[graph.batch[avail] == g_idx]
                    if len(avail_g) == 0:
                        continue
                    if policy_logits.shape[0] == graph.num_edges:
                        raw_g = policy_logits[avail_g]
                    else:
                        raw_g = policy_logits[g_idx][:len(avail_g)]
                    raw_g = F.softmax(raw_g, dim=0)
                    probs_all[avail_g] = raw_g

                if probs_all.sum() > 0:
                    probs_all = probs_all / (probs_all.sum() + EPS)
                else:
                    probs_all[avail] = 1.0 / len(avail)

                a = node.untried_actions.pop(0)
                child_state = node.state.clone()
                child_state[a] = True
                child = MCTSNode(child_state, parent=node, prior=float(probs_all[a].item()))
                node.children[a] = child
                node = child
                path.append(node)

            # Simulation / Rollout
            v = rollout_policy(node, graph, rc_explainer, model, max_budget, debias_flag, pred_bias_list, rollout_limit)

        # Backpropagate
        for n in path:
            n.N += 1
            n.W += v
            n.Q = n.W / (n.N + 0.0)

    # Select best action
    if len(root.children) == 0:
        with torch.no_grad():
            out = rc_explainer(graph, root.state, train_flag=False)
        policy_logits = out[1] if len(out) >= 2 else out[0]
        action = int(torch.argmax(policy_logits).item())
        visit_dist = None
    else:
        visit_counts = {a: child.N for a, child in root.children.items()}
        total = sum(visit_counts.values()) + EPS
        visit_dist = {a: cnt / total for a, cnt in visit_counts.items()}
        action = max(visit_counts.items(), key=lambda x: x[1])[0]

    return int(action), visit_dist


# -----------------------------
# Bias detector, batch support
# -----------------------------
def bias_detector(model, graph, valid_budget):
    batch_size = graph.y.size(0)
    pred_bias_list = []

    for budget in range(valid_budget):
        num_repeat = 2
        i_pred_bias = 0.
        for i in range(num_repeat):
            bias_selection = torch.zeros(graph.num_edges, dtype=torch.bool, device=device)
            ava_action_batch = graph.batch[graph.edge_index[0]]
            ava_action_probs = torch.rand(ava_action_batch.size(), device=device)
            _, added_actions = scatter_max(ava_action_probs, ava_action_batch)

            for g_idx in range(batch_size):
                bias_selection[graph.batch == g_idx][added_actions[g_idx]] = True

            bias_subgraph = relabel_graph(graph, bias_selection)
            bias_subgraph_pred = model(bias_subgraph.x, bias_subgraph.edge_index,
                                       bias_subgraph.edge_attr, bias_subgraph.batch).detach()
            i_pred_bias += bias_subgraph_pred / num_repeat
        pred_bias_list.append(i_pred_bias)

    return pred_bias_list

def get_reward(full_pred, subgraph_pred, target_y, pre_reward=None):



    
    device = next(full_pred.parameters()).device if hasattr(full_pred, 'parameters') else full_pred.device

    full_pred = full_pred.to(device)
    subgraph_pred = subgraph_pred.to(device)
    target_y = target_y.to(device)
    if pre_reward is None:
        pre_reward = torch.zeros(target_y.size(), device=device)
    else:
        pre_reward = pre_reward.to(device)

    reward = pre_reward.clone()

    # 你的 reward 计算逻辑
    reward += 2 * (target_y == subgraph_pred.argmax(dim=1)).float() - 1.
    reward += 0.97 * pre_reward

    return reward

def train_policy_mcts_ppo(rc_explainer, model, train_loader, test_loader, optimizer,
                          topK_ratio=0.1, debias_flag=False, topN=None, batch_size=32,
                          num_epochs=30, num_simulations=20, c_puct=1.0, clip_ratio=0.2, gamma=0.97,
                          rollout_limit=None, ppo_epochs=4):
    best_acc_auc = best_acc_curve = best_pre = best_rec = 0

    for ep in range(num_epochs):
        rc_explainer.train()
        model.eval()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{num_epochs}", ncols=120)
        for graph in pbar:
            graph = graph.to(device)
            max_budget = max(int(topK_ratio * graph.num_edges / batch_size), 1)

            # optional bias correction
            pred_bias_list = bias_detector(model, graph, max_budget) if debias_flag else None

            # PPO storage
            traj_log_probs = []
            traj_actions = []
            traj_rewards = []

            # current state mask per batch graph
            current_state = torch.zeros(graph.num_edges, dtype=torch.bool, device=device)

            # for each step in budget
            for budget_step in range(max_budget):
                # batch-wise MCTS: run MCTS for each graph in the batch
                actions = []
                for g_idx in range(batch_size):
                    # mask edges for this graph
                    g_mask = graph.batch == g_idx
                    state_g = current_state.clone()
                    state_g[~g_mask] = True  # mark other graph edges as done
                    action_g, _ = mcts_search(state_g, graph, rc_explainer, model,
                                              max_budget, num_simulations=num_simulations,
                                              c_puct=c_puct, debias_flag=debias_flag,
                                              pred_bias_list=pred_bias_list,
                                              rollout_limit=rollout_limit)
                    actions.append(action_g)
                    current_state[action_g] = True

                # compute log_probs and rewards for PPO
                log_probs_batch = []
                rewards_batch = []
                for g_idx, a in enumerate(actions):
                    g_mask = graph.batch == g_idx
                    # reconstruct previous state for log_prob
                    prev_state = current_state.clone()
                    prev_state[a] = False
                    out_prev = rc_explainer(graph, prev_state, train_flag=False)
                    logits_prev = out_prev[1] if len(out_prev) >= 2 else out_prev[0]
                    if logits_prev.shape[0] == graph.num_edges:
                        probs_prev = F.softmax(logits_prev, dim=0)
                        log_prob = torch.log(probs_prev[a] + EPS)
                    else:
                        avail_prev = (~prev_state).nonzero(as_tuple=False).flatten()
                        pos = (avail_prev == a).nonzero(as_tuple=False).item()
                        raw_prev = F.softmax(logits_prev[g_idx][:len(avail_prev)], dim=0)
                        log_prob = torch.log(raw_prev[pos] + EPS)
                    log_probs_batch.append(log_prob)

                # compute rewards after this budget step
                subgraph = relabel_graph(graph, current_state)
                subgraph_pred = model(subgraph.x, subgraph.edge_index, subgraph.edge_attr, subgraph.batch)
                if debias_flag:
                    for g_idx in range(batch_size):
                        idx = int(current_state[graph.batch == g_idx].sum().item()) - 1
                        subgraph_pred[graph.batch == g_idx] -= pred_bias_list[idx][g_idx]
                    subgraph_pred = F.softmax(subgraph_pred.detach())
                else:
                    subgraph_pred = F.softmax(subgraph_pred.detach())
                full_pred = F.softmax(model(graph.x, graph.edge_index, graph.edge_attr, graph.batch).detach())
                reward_vec = get_reward(full_pred, subgraph_pred, graph.y, pre_reward=torch.zeros(graph.y.size(), device=device))
                reward = float(reward_vec.mean().item())

                # store PPO trajectory
                traj_log_probs.extend(log_probs_batch)
                traj_actions.extend(actions)
                traj_rewards.extend([reward] * batch_size)  # same reward per graph

            # --- PPO update ---
            returns = []
            R = 0.0
            for r in reversed(traj_rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32, device=device)
            returns = (returns - returns.mean()) / (returns.std() + EPS)

            if len(traj_log_probs) == 0:
                continue
            old_log_probs = torch.stack(traj_log_probs).detach()
            for _ in range(ppo_epochs):
                current_log_probs = []
                for t, a in enumerate(traj_actions):
                    state_t = torch.zeros(graph.num_edges, dtype=torch.bool, device=device)
                    for aa in traj_actions[:t]:
                        state_t[aa] = True
                    out = rc_explainer(graph, state_t, train_flag=True)
                    logits = out[1] if len(out) >= 2 else out[0]
                    if logits.shape[0] == graph.num_edges:
                        probs = F.softmax(logits, dim=0)
                        lp = torch.log(probs[a] + EPS)
                    else:
                        avail = (~state_t).nonzero(as_tuple=False).flatten()
                        raw = F.softmax(logits[0][:len(avail)], dim=0)  # assume single graph logits
                        pos = (avail == a).nonzero(as_tuple=False).item()
                        lp = torch.log(raw[pos] + EPS)
                    current_log_probs.append(lp)
                current_log_probs = torch.stack(current_log_probs)

                advantages = returns.detach()
                ratio = torch.exp(current_log_probs - old_log_probs)
                clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
                loss = -torch.mean(torch.min(ratio * advantages, clipped))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())

            pbar.set_postfix(loss=total_loss)

        # --- evaluation ---
        ep_acc_auc, ep_acc_curve, ep_pre, ep_rec = test_policy_all_with_gnd(rc_explainer, model, test_loader, topN)
        print(f"[EP {ep+1}] TotalLoss={total_loss:.4f}, ACC-AUC={ep_acc_auc:.4f}, Pre@{topN}={ep_pre:.4f}, Rec@{topN}={ep_rec:.4f}")

        # save best
        if ep_acc_auc >= best_acc_auc:
            best_acc_auc = ep_acc_auc
            best_acc_curve = ep_acc_curve
            best_pre = ep_pre
            best_rec = ep_rec
            rc_explainer.save_policy_net(path=None)

    return rc_explainer, best_acc_auc, best_acc_curve, best_pre, best_rec


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

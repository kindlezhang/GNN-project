# Overview

Causal explanation for Graph Neural Networks (GNNs) is an emerging research direction aimed at uncovering the underlying causal subgraph that drives GNN predictions, overcoming limitations of prior association-based methods prone to spurious correlations. This study proposes to advance GNN explanations from a causal perspective by integrating causal inference principles and reinforcement learning methods, inspired by recent works such as the Reinforced Causal Explainer and Graph Neural Network Causal Explanation via Neural Causal Models (CXGNN). The goal is to develop novel, interpretable, and mathematically grounded models that can identify subgraphs with true causal influence on GNN outputs.

# Objectives

**The objectives of this study include:**

- Developing a causal explanation framework for GNNs that leverages do-calculus, Individual Conditional Expectation (ICE), and structural causal models (SCMs) to quantify causal effects of edges and nodes on final predictions.

- Formulating the explanation problem as a decision-making or multi-objective optimization problem inspired by reinforcement learning. In particular, applying Q-learning with Bellman equations to guide subgraph extraction by maximizing causal reward signals.

- Alternatively, revisiting the SCM-based approach from a causal inference perspective to construct neural causal models (NCMs) customized for GNNs, allowing efficient estimation of causal effects in complex graphs.

- Evaluating the proposed methods on classical benchmark datasets for GNN explanation, such as MUTAG, PROTEINS, and BA-2Motifs, to validate interpretability and faithfulness of the causal subgraph explanations.

# Data Sources

The study will utilize standard GNN benchmark datasets widely used for interpretability and causal explanation studies, including:

- **MUTAG:** A dataset of chemical compounds commonly used for graph classification and explanation.

- **PROTEINS:** A protein structure dataset featuring graphs representing protein domains.

- **BA-2Motifs:** A synthetic graph dataset designed to test graph explanation algorithms in identifying relevant motifs.

These datasets enable fair comparison with existing methods and offer ground-truth or domain-informed subgraphs for quantitative evaluation.


# Steps in the Study

1. **Literature Review and Theory Establishment:**
   
   - Deep dive into causal inference basics, including do-calculus, SCMs, and neural causal models. Analyze reinforcement learning fundamentals, focusing on Q-learning and Bellman equations applied to edge selection as actions.

2. **Method Development:**

   - For the first approach, adapt the reinforced causal explainer architecture by incorporating Q-learning to treat edge selection as a stepwise decision process, where rewards correspond to causal effect estimates leading to interpretable subgraph construction.
   - For the second approach, leverage SCM frameworks from causal inference coursework to formalize a GNN-specific SCM, and instantiate neural causal models to efficiently estimate causal interactions and pinpoint explanatory subgraphs.

3. **Implementation:**

   - Develop computational pipelines implementing both methods using available graph ML toolkits and reinforcement learning frameworks.
  
4. **Experimental Evaluation:**

   - Conduct experiments on the selected datasets, comparing the proposed causal explainers with baseline association-based and existing causal methods. Metrics include explanation fidelity, subgraph size, precision, and recall against ground-truth explanations.

5. **Analysis and Interpretation:**

   - Analyze quantitative results and visualize explanatory subgraphs. Assess how well each method disentangles true causal effects from spurious correlations.
  

# Expected Outcomes

The study expects to deliver:

- A novel reinforcement learning based causal explanation model for GNNs that constructs causal subgraphs via optimized multi-step decision making.

- An SCM and neural causal model-based causal explainer that directly quantifies causal relations in graphs for GNN predictions.

- Empirical evidence that causal explainers outperform association-based methods, with higher precision in identifying true causal subgraphs and robustness across datasets.

- Insights into the advantages and limitations of reinforcement learning versus SCM-based causal explanation paradigms for GNNs.

# Importance of the Study

This study contributes to explainable AI by advancing causal interpretability of Graph Neural Networks, a class of powerful models widely applied in chemistry, biology, social networks, and recommendation systems. By moving beyond correlation towards causal understanding, the proposed framework improves reliability and trustworthiness of GNN decisions, enabling practitioners to identify truly influential graph components. The integration of reinforcement learning and causal inference also opens new avenues to tackle a complex combinatorial search space of subgraphs, making the explanations both rigorous and computationally feasible. Overall, this research lays foundational work for future causal explainability studies and practical applications in critical domains.
 
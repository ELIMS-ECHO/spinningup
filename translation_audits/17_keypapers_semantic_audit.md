# 第17篇《Key Papers in Deep RL》逐条翻译与 Sphinx 验收记录

## 原始版本和交付版本

- 上游原文：`openai/spinningup/docs/spinningup/keypapers.rst`；文件 blob `161416a3525941fda5b38713e4720a7fbea63971`。
- 中文译文：`tianyueni78-cyber/spinningup` 的 `learning-edition-zh` 分支，`docs/spinningup/keypapers_zh.rst`；审计时的文件 blob `0f254d10a60d9895fb2acecfa5d02f2413bbbe2e`。
- 首次全文翻译 commit：`b91a753744e564796fe68605fc4ece55418637c6`。
- 术语修正：`252f54c33ca68af2ce2385eb08a7d486228d933a`，将 Distributional RL 明确译为“分布强化学习”，不是“分布式强化学习”。
- 格式修正：`fd519a9a2e065a4567702a58283c146793c54a44`，为 14 项贡献说明补齐 RST 加粗结束标记。

## 审计方法

按文件的原始行序逐条检查全部 105 个 `.. [#]` 条目；对每条检查：英文论文标题、外链、作者、年份、标点和所在分类是否保留；如带 `Algorithm` 标签，再严格比较算法字符串是否一致；如带 `Contribution(s)`，逐条人工检查贡献说明是否忠实于英文。没有说明的条目保持只有题录，不能增写未经原文支持的贡献。
正文导语、目录深度和所有分类标题另行对照。此文是截至原文版本为止的历史论文清单，不更新作者年份或用后续新论文替换它。论文英文题名、署名、年份保留原样，便于追溯和查找原始文献。

## 原文目录与中文分类逐项对应（26 项）

| 原文行号 | 英文标题 | 中文标题 | 检查结果 |
|---:|---|---|---|
| 2 | Key Papers in Deep RL | 深度强化学习关键论文 | 一致 |
| 11 | 1. Model-Free RL | 1. 无模型强化学习 | 一致 |
| 14 | a. Deep Q-Learning | a. 深度 Q 学习 | 一致 |
| 31 | b. Policy Gradients | b. 策略梯度 | 一致 |
| 51 | c. Deterministic Policy Gradients | c. 确定性策略梯度 | 一致 |
| 62 | d. Distributional RL | d. 分布强化学习 | 一致 |
| 73 | e. Policy Gradients with Action-Dependent Baselines | e. 使用依赖动作基线的策略梯度 | 一致 |
| 83 | f. Path-Consistency Learning | f. 路径一致性学习 | 一致 |
| 90 | g. Other Directions for Combining Policy-Learning and Q-Learning | g. 结合策略学习与 Q 学习的其他方向 | 一致 |
| 102 | h. Evolutionary Algorithms | h. 进化算法 | 一致 |
| 109 | 2. Exploration | 2. 探索 | 一致 |
| 112 | a. Intrinsic Motivation | a. 内在动机 | 一致 |
| 132 | b. Unsupervised RL | b. 无监督强化学习 | 一致 |
| 142 | 3. Transfer and Multitask RL | 3. 迁移与多任务强化学习 | 一致 |
| 161 | 4. Hierarchy | 4. 层次化强化学习 | 一致 |
| 170 | 5. Memory | 5. 记忆 | 一致 |
| 184 | 6. Model-Based RL | 6. 基于模型的强化学习 | 一致 |
| 187 | a. Model is Learned | a. 模型通过学习获得 | 一致 |
| 204 | b. Model is Given | b. 模型已知 | 一致 |
| 211 | 7. Meta-RL | 7. 元强化学习 | 一致 |
| 222 | 8. Scaling RL | 8. 强化学习的规模扩展 | 一致 |
| 236 | 9. RL in the Real World | 9. 现实世界中的强化学习 | 一致 |
| 248 | 10. Safety | 10. 安全性 | 一致 |
| 264 | 11. Imitation Learning and Inverse Reinforcement Learning | 11. 模仿学习与逆强化学习 | 一致 |
| 280 | 12. Reproducibility, Analysis, and Critique | 12. 可复现性、分析与批判 | 一致 |
| 297 | 13. Bonus: Classic Papers in RL Theory or Review | 13. 补充：经典强化学习理论与综述论文 | 一致 |

## 13 个顶层主题的条目数量核对

| 顶层分类（原文） | 论文数量 |
|---|---:|
| 1. Model-Free RL | 31 |
| 2. Exploration | 11 |
| 3. Transfer and Multitask RL | 8 |
| 4. Hierarchy | 3 |
| 5. Memory | 5 |
| 6. Model-Based RL | 9 |
| 7. Meta-RL | 4 |
| 8. Scaling RL | 5 |
| 9. RL in the Real World | 4 |
| 10. Safety | 6 |
| 11. Imitation Learning and Inverse Reinforcement Learning | 6 |
| 12. Reproducibility, Analysis, and Critique | 7 |
| 13. Bonus: Classic Papers in RL Theory or Review | 6 |

## 105 篇论文逐条元数据与标签核对

每条论文在原文与译文中的行号完全相同。对每条论文，严格检查题名及链接构成的原始 RST 超链接串、作者和出版年份文本；算法名称完全一致，贡献类另在下一节核对具体语义。

| 顺序 | 两版行号 | 所属原文分类 | 原始论文题名 | 说明类型 | 核对结果 |
|---:|---:|---|---|---|---|
| 1 | 18 | 1. Model-Free RL / a. Deep Q-Learning | Playing Atari with Deep Reinforcement Learning | 算法 | 元数据与标签通过 |
| 2 | 20 | 1. Model-Free RL / a. Deep Q-Learning | Deep Recurrent Q-Learning for Partially Observable MDPs | 算法 | 元数据与标签通过 |
| 3 | 22 | 1. Model-Free RL / a. Deep Q-Learning | Dueling Network Architectures for Deep Reinforcement Learning | 算法 | 元数据与标签通过 |
| 4 | 24 | 1. Model-Free RL / a. Deep Q-Learning | Deep Reinforcement Learning with Double Q-learning | 算法 | 元数据与标签通过 |
| 5 | 26 | 1. Model-Free RL / a. Deep Q-Learning | Prioritized Experience Replay | 算法 | 元数据与标签通过 |
| 6 | 28 | 1. Model-Free RL / a. Deep Q-Learning | Rainbow: Combining Improvements in Deep Reinforcement Learning | 算法 | 元数据与标签通过 |
| 7 | 35 | 1. Model-Free RL / b. Policy Gradients | Asynchronous Methods for Deep Reinforcement Learning | 算法 | 元数据与标签通过 |
| 8 | 37 | 1. Model-Free RL / b. Policy Gradients | Trust Region Policy Optimization | 算法 | 元数据与标签通过 |
| 9 | 39 | 1. Model-Free RL / b. Policy Gradients | High-Dimensional Continuous Control Using Generalized Advantage Estimation | 算法 | 元数据与标签通过 |
| 10 | 41 | 1. Model-Free RL / b. Policy Gradients | Proximal Policy Optimization Algorithms | 算法 | 元数据与标签通过 |
| 11 | 43 | 1. Model-Free RL / b. Policy Gradients | Emergence of Locomotion Behaviours in Rich Environments | 算法 | 元数据与标签通过 |
| 12 | 45 | 1. Model-Free RL / b. Policy Gradients | Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation | 算法 | 元数据与标签通过 |
| 13 | 47 | 1. Model-Free RL / b. Policy Gradients | Sample Efficient Actor-Critic with Experience Replay | 算法 | 元数据与标签通过 |
| 14 | 49 | 1. Model-Free RL / b. Policy Gradients | Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor | 算法 | 元数据与标签通过 |
| 15 | 55 | 1. Model-Free RL / c. Deterministic Policy Gradients | Deterministic Policy Gradient Algorithms | 算法 | 元数据与标签通过 |
| 16 | 57 | 1. Model-Free RL / c. Deterministic Policy Gradients | Continuous Control With Deep Reinforcement Learning | 算法 | 元数据与标签通过 |
| 17 | 59 | 1. Model-Free RL / c. Deterministic Policy Gradients | Addressing Function Approximation Error in Actor-Critic Methods | 算法 | 元数据与标签通过 |
| 18 | 65 | 1. Model-Free RL / d. Distributional RL | A Distributional Perspective on Reinforcement Learning | 算法 | 元数据与标签通过 |
| 19 | 67 | 1. Model-Free RL / d. Distributional RL | Distributional Reinforcement Learning with Quantile Regression | 算法 | 元数据与标签通过 |
| 20 | 69 | 1. Model-Free RL / d. Distributional RL | Implicit Quantile Networks for Distributional Reinforcement Learning | 算法 | 元数据与标签通过 |
| 21 | 71 | 1. Model-Free RL / d. Distributional RL | Dopamine: A Research Framework for Deep Reinforcement Learning | 贡献 | 元数据与标签通过 |
| 22 | 76 | 1. Model-Free RL / e. Policy Gradients with Action-Dependent Baselines | Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic | 算法 | 元数据与标签通过 |
| 23 | 78 | 1. Model-Free RL / e. Policy Gradients with Action-Dependent Baselines | Action-depedent Control Variates for Policy Optimization via Stein's Identity | 算法 | 元数据与标签通过 |
| 24 | 80 | 1. Model-Free RL / e. Policy Gradients with Action-Dependent Baselines | The Mirage of Action-Dependent Baselines in Reinforcement Learning | 贡献 | 元数据与标签通过 |
| 25 | 86 | 1. Model-Free RL / f. Path-Consistency Learning | Bridging the Gap Between Value and Policy Based Reinforcement Learning | 算法 | 元数据与标签通过 |
| 26 | 88 | 1. Model-Free RL / f. Path-Consistency Learning | Trust-PCL: An Off-Policy Trust Region Method for Continuous Control | 算法 | 元数据与标签通过 |
| 27 | 93 | 1. Model-Free RL / g. Other Directions for Combining Policy-Learning and Q-Learning | Combining Policy Gradient and Q-learning | 算法 | 元数据与标签通过 |
| 28 | 95 | 1. Model-Free RL / g. Other Directions for Combining Policy-Learning and Q-Learning | The Reactor: A Fast and Sample-Efficient Actor-Critic Agent for Reinforcement Learning | 算法 | 元数据与标签通过 |
| 29 | 97 | 1. Model-Free RL / g. Other Directions for Combining Policy-Learning and Q-Learning | Interpolated Policy Gradient: Merging On-Policy and Off-Policy Gradient Estimation for Deep Reinforcement Learning | 算法 | 元数据与标签通过 |
| 30 | 99 | 1. Model-Free RL / g. Other Directions for Combining Policy-Learning and Q-Learning | Equivalence Between Policy Gradients and Soft Q-Learning | 贡献 | 元数据与标签通过 |
| 31 | 105 | 1. Model-Free RL / h. Evolutionary Algorithms | Evolution Strategies as a Scalable Alternative to Reinforcement Learning | 算法 | 元数据与标签通过 |
| 32 | 115 | 2. Exploration / a. Intrinsic Motivation | VIME: Variational Information Maximizing Exploration | 算法 | 元数据与标签通过 |
| 33 | 117 | 2. Exploration / a. Intrinsic Motivation | Unifying Count-Based Exploration and Intrinsic Motivation | 算法 | 元数据与标签通过 |
| 34 | 119 | 2. Exploration / a. Intrinsic Motivation | Count-Based Exploration with Neural Density Models | 算法 | 元数据与标签通过 |
| 35 | 121 | 2. Exploration / a. Intrinsic Motivation | #Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning | 算法 | 元数据与标签通过 |
| 36 | 123 | 2. Exploration / a. Intrinsic Motivation | EX2: Exploration with Exemplar Models for Deep Reinforcement Learning | 算法 | 元数据与标签通过 |
| 37 | 125 | 2. Exploration / a. Intrinsic Motivation | Curiosity-driven Exploration by Self-supervised Prediction | 算法 | 元数据与标签通过 |
| 38 | 127 | 2. Exploration / a. Intrinsic Motivation | Large-Scale Study of Curiosity-Driven Learning | 贡献 | 元数据与标签通过 |
| 39 | 129 | 2. Exploration / a. Intrinsic Motivation | Exploration by Random Network Distillation | 算法 | 元数据与标签通过 |
| 40 | 135 | 2. Exploration / b. Unsupervised RL | Variational Intrinsic Control | 算法 | 元数据与标签通过 |
| 41 | 137 | 2. Exploration / b. Unsupervised RL | Diversity is All You Need: Learning Skills without a Reward Function | 算法 | 元数据与标签通过 |
| 42 | 139 | 2. Exploration / b. Unsupervised RL | Variational Option Discovery Algorithms | 算法 | 元数据与标签通过 |
| 43 | 145 | 3. Transfer and Multitask RL / 无子类 | Progressive Neural Networks | 算法 | 元数据与标签通过 |
| 44 | 147 | 3. Transfer and Multitask RL / 无子类 | Universal Value Function Approximators | 算法 | 元数据与标签通过 |
| 45 | 149 | 3. Transfer and Multitask RL / 无子类 | Reinforcement Learning with Unsupervised Auxiliary Tasks | 算法 | 元数据与标签通过 |
| 46 | 151 | 3. Transfer and Multitask RL / 无子类 | The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously | 算法 | 元数据与标签通过 |
| 47 | 153 | 3. Transfer and Multitask RL / 无子类 | PathNet: Evolution Channels Gradient Descent in Super Neural Networks | 算法 | 元数据与标签通过 |
| 48 | 155 | 3. Transfer and Multitask RL / 无子类 | Mutual Alignment Transfer Learning | 算法 | 元数据与标签通过 |
| 49 | 157 | 3. Transfer and Multitask RL / 无子类 | Learning an Embedding Space for Transferable Robot Skills | 仅题录 | 元数据与标签通过 |
| 50 | 159 | 3. Transfer and Multitask RL / 无子类 | Hindsight Experience Replay | 算法 | 元数据与标签通过 |
| 51 | 164 | 4. Hierarchy / 无子类 | Strategic Attentive Writer for Learning Macro-Actions | 算法 | 元数据与标签通过 |
| 52 | 166 | 4. Hierarchy / 无子类 | FeUdal Networks for Hierarchical Reinforcement Learning | 算法 | 元数据与标签通过 |
| 53 | 168 | 4. Hierarchy / 无子类 | Data-Efficient Hierarchical Reinforcement Learning | 算法 | 元数据与标签通过 |
| 54 | 173 | 5. Memory / 无子类 | Model-Free Episodic Control | 算法 | 元数据与标签通过 |
| 55 | 176 | 5. Memory / 无子类 | Neural Episodic Control | 算法 | 元数据与标签通过 |
| 56 | 178 | 5. Memory / 无子类 | Neural Map: Structured Memory for Deep Reinforcement Learning | 算法 | 元数据与标签通过 |
| 57 | 180 | 5. Memory / 无子类 | Unsupervised Predictive Memory in a Goal-Directed Agent | 算法 | 元数据与标签通过 |
| 58 | 182 | 5. Memory / 无子类 | Relational Recurrent Neural Networks | 算法 | 元数据与标签通过 |
| 59 | 190 | 6. Model-Based RL / a. Model is Learned | Imagination-Augmented Agents for Deep Reinforcement Learning | 算法 | 元数据与标签通过 |
| 60 | 192 | 6. Model-Based RL / a. Model is Learned | Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning | 算法 | 元数据与标签通过 |
| 61 | 194 | 6. Model-Based RL / a. Model is Learned | Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning | 算法 | 元数据与标签通过 |
| 62 | 196 | 6. Model-Based RL / a. Model is Learned | Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion | 算法 | 元数据与标签通过 |
| 63 | 198 | 6. Model-Based RL / a. Model is Learned | Model-Ensemble Trust-Region Policy Optimization | 算法 | 元数据与标签通过 |
| 64 | 200 | 6. Model-Based RL / a. Model is Learned | Model-Based Reinforcement Learning via Meta-Policy Optimization | 算法 | 元数据与标签通过 |
| 65 | 202 | 6. Model-Based RL / a. Model is Learned | Recurrent World Models Facilitate Policy Evolution | 仅题录 | 元数据与标签通过 |
| 66 | 207 | 6. Model-Based RL / b. Model is Given | Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm | 算法 | 元数据与标签通过 |
| 67 | 209 | 6. Model-Based RL / b. Model is Given | Thinking Fast and Slow with Deep Learning and Tree Search | 算法 | 元数据与标签通过 |
| 68 | 214 | 7. Meta-RL / 无子类 | RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning | 算法 | 元数据与标签通过 |
| 69 | 216 | 7. Meta-RL / 无子类 | Learning to Reinforcement Learn | 仅题录 | 元数据与标签通过 |
| 70 | 218 | 7. Meta-RL / 无子类 | Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks | 算法 | 元数据与标签通过 |
| 71 | 220 | 7. Meta-RL / 无子类 | A Simple Neural Attentive Meta-Learner | 算法 | 元数据与标签通过 |
| 72 | 225 | 8. Scaling RL / 无子类 | Accelerated Methods for Deep Reinforcement Learning | 贡献 | 元数据与标签通过 |
| 73 | 227 | 8. Scaling RL / 无子类 | IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures | 算法 | 元数据与标签通过 |
| 74 | 229 | 8. Scaling RL / 无子类 | Distributed Prioritized Experience Replay | 算法 | 元数据与标签通过 |
| 75 | 231 | 8. Scaling RL / 无子类 | Recurrent Experience Replay in Distributed Reinforcement Learning | 算法 | 元数据与标签通过 |
| 76 | 233 | 8. Scaling RL / 无子类 | RLlib: Abstractions for Distributed Reinforcement Learning | 贡献 | 元数据与标签通过 |
| 77 | 239 | 9. RL in the Real World / 无子类 | Benchmarking Reinforcement Learning Algorithms on Real-World Robots | 仅题录 | 元数据与标签通过 |
| 78 | 241 | 9. RL in the Real World / 无子类 | Learning Dexterous In-Hand Manipulation | 仅题录 | 元数据与标签通过 |
| 79 | 243 | 9. RL in the Real World / 无子类 | QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation | 算法 | 元数据与标签通过 |
| 80 | 245 | 9. RL in the Real World / 无子类 | Horizon: Facebook's Open Source Applied Reinforcement Learning Platform | 仅题录 | 元数据与标签通过 |
| 81 | 251 | 10. Safety / 无子类 | Concrete Problems in AI Safety | 贡献 | 元数据与标签通过 |
| 82 | 253 | 10. Safety / 无子类 | Deep Reinforcement Learning From Human Preferences | 算法 | 元数据与标签通过 |
| 83 | 255 | 10. Safety / 无子类 | Constrained Policy Optimization | 算法 | 元数据与标签通过 |
| 84 | 257 | 10. Safety / 无子类 | Safe Exploration in Continuous Action Spaces | 算法 | 元数据与标签通过 |
| 85 | 259 | 10. Safety / 无子类 | Trial without Error: Towards Safe Reinforcement Learning via Human Intervention | 算法 | 元数据与标签通过 |
| 86 | 261 | 10. Safety / 无子类 | Leave No Trace: Learning to Reset for Safe and Autonomous Reinforcement Learning | 算法 | 元数据与标签通过 |
| 87 | 267 | 11. Imitation Learning and Inverse Reinforcement Learning / 无子类 | Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy | 贡献 | 元数据与标签通过 |
| 88 | 269 | 11. Imitation Learning and Inverse Reinforcement Learning / 无子类 | Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization | 算法 | 元数据与标签通过 |
| 89 | 271 | 11. Imitation Learning and Inverse Reinforcement Learning / 无子类 | Generative Adversarial Imitation Learning | 算法 | 元数据与标签通过 |
| 90 | 273 | 11. Imitation Learning and Inverse Reinforcement Learning / 无子类 | DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills | 算法 | 元数据与标签通过 |
| 91 | 275 | 11. Imitation Learning and Inverse Reinforcement Learning / 无子类 | Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow | 算法 | 元数据与标签通过 |
| 92 | 277 | 11. Imitation Learning and Inverse Reinforcement Learning / 无子类 | One-Shot High-Fidelity Imitation: Training Large-Scale Deep Nets with RL | 算法 | 元数据与标签通过 |
| 93 | 283 | 12. Reproducibility, Analysis, and Critique / 无子类 | Benchmarking Deep Reinforcement Learning for Continuous Control | 贡献 | 元数据与标签通过 |
| 94 | 285 | 12. Reproducibility, Analysis, and Critique / 无子类 | Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control | 仅题录 | 元数据与标签通过 |
| 95 | 287 | 12. Reproducibility, Analysis, and Critique / 无子类 | Deep Reinforcement Learning that Matters | 仅题录 | 元数据与标签通过 |
| 96 | 289 | 12. Reproducibility, Analysis, and Critique / 无子类 | Where Did My Optimum Go?: An Empirical Analysis of Gradient Descent Optimization in Policy Gradient Methods | 仅题录 | 元数据与标签通过 |
| 97 | 291 | 12. Reproducibility, Analysis, and Critique / 无子类 | Are Deep Policy Gradient Algorithms Truly Policy Gradient Algorithms? | 仅题录 | 元数据与标签通过 |
| 98 | 293 | 12. Reproducibility, Analysis, and Critique / 无子类 | Simple Random Search Provides a Competitive Approach to Reinforcement Learning | 仅题录 | 元数据与标签通过 |
| 99 | 295 | 12. Reproducibility, Analysis, and Critique / 无子类 | Benchmarking Model-Based Reinforcement Learning | 仅题录 | 元数据与标签通过 |
| 100 | 300 | 13. Bonus: Classic Papers in RL Theory or Review / 无子类 | Policy Gradient Methods for Reinforcement Learning with Function Approximation | 贡献 | 元数据与标签通过 |
| 101 | 302 | 13. Bonus: Classic Papers in RL Theory or Review / 无子类 | An Analysis of Temporal-Difference Learning with Function Approximation | 贡献 | 元数据与标签通过 |
| 102 | 304 | 13. Bonus: Classic Papers in RL Theory or Review / 无子类 | Reinforcement Learning of Motor Skills with Policy Gradients | 贡献 | 元数据与标签通过 |
| 103 | 306 | 13. Bonus: Classic Papers in RL Theory or Review / 无子类 | Approximately Optimal Approximate Reinforcement Learning | 贡献 | 元数据与标签通过 |
| 104 | 308 | 13. Bonus: Classic Papers in RL Theory or Review / 无子类 | A Natural Policy Gradient | 贡献 | 元数据与标签通过 |
| 105 | 310 | 13. Bonus: Classic Papers in RL Theory or Review / 无子类 | Algorithms for Reinforcement Learning | 贡献 | 元数据与标签通过 |

## 15 条论文贡献说明的独立语义核对

| 论文序号 | 两版行号 | 英文论文题名 | 中文必须表达的原文含义 | 结果 |
|---:|---:|---|---|---|
| 21 | 71 | Dopamine: A Research Framework for Deep Reinforcement Learning | 介绍包含 DQN、C51、IQN、Rainbow 实现的 Dopamine 代码仓库，代码链接保持不变 | 通过 |
| 24 | 80 | The Mirage of Action-Dependent Baselines in Reinforcement Learning | 批评并重新评估 Q-Prop、Stein 控制变量等早期论文的主张，发现重要方法学错误 | 通过 |
| 30 | 99 | Equivalence Between Policy Gradients and Soft Q-Learning | 揭示策略梯度与软 Q 学习两类方法的理论联系 | 通过 |
| 38 | 127 | Large-Scale Study of Curiosity-Driven Learning | 系统评估基于惊讶度的内在动机在广泛环境中的表现 | 通过 |
| 72 | 225 | Accelerated Methods for Deep Reinforcement Learning | 系统分析跨不同深度 RL 算法的并行化 | 通过 |
| 76 | 233 | RLlib: Abstractions for Distributed Reinforcement Learning | 介绍可扩展的 RL 实现库，并完整保留文档链接 | 通过 |
| 81 | 251 | Concrete Problems in AI Safety | 建立安全问题分类体系，是后续研究的重要起点，并保留应解决这些问题的呼吁 | 通过 |
| 87 | 267 | Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy | 清晰形式化最大熵逆强化学习 IRL | 通过 |
| 93 | 283 | Benchmarking Deep Reinforcement Learning for Continuous Control | 贡献名称 rllab 原样保留 | 通过 |
| 100 | 300 | Policy Gradient Methods for Reinforcement Learning with Function Approximation | 建立策略梯度定理并证明任意策略类下策略梯度算法收敛 | 通过 |
| 101 | 302 | An Analysis of Temporal-Difference Learning with Function Approximation | 关于强化学习价值学习方法的收敛性结果与反例 | 通过 |
| 102 | 304 | Reinforcement Learning of Motor Skills with Policy Gradients | 系统回顾当时的策略梯度方法，其中部分描述仍适用于深度 RL | 通过 |
| 103 | 306 | Approximately Optimal Approximate Reinforcement Learning | 单调改进理论早期根源，后来支持 TRPO 等方法的理论解释 | 通过 |
| 104 | 308 | A Natural Policy Gradient | 把自然梯度引入 RL，进而推动 TRPO、ACKTR 等方法 | 通过 |
| 105 | 310 | Algorithms for Reinforcement Learning | 深度 RL 发展之前的 RL 基础与理论综述，原文称其为卓越参考资料 | 通过 |

## 程序化数据完整性结果

- 论文条目总数：上游 105，译文 105；两版每条对应的行号、英文标题、作者/年份题录文本顺序逐一比对，差异 0。
- 算法标签：78 条，全部算法名逐字符一致（仅将字段名称 `Algorithm` 翻译为“算法”）。
- 贡献说明：15 条，已进行中英文逐项语义核对，并修复 RST 格式。
- 不含算法或贡献说明的题录：12 条；完全按原文保留，不新增推断性描述。
- 原文外部 HTTP(S) 链接：107 个匹配项；译文链接序列与原文逐个完全相同，遗漏、改写和重排均为 0。
- 分类标题：26 项，译文分类与论文顺序一一对应。
- 导语：原文强调论文清单“远非全面”，仅用于研究起点；译文保留这一限制。目录指令 `.. contents::` 及 `:depth: 2` 保留。

## 实际 Sphinx HTML 构建

- 失败记录：https://github.com/tianyueni78-cyber/spinningup/actions/runs/38050681789 。旧版存在 14 处未闭合的 `**贡献：` 加粗语法，导致 `Inline strong start-string without end-string` 警告。
- 已提交格式修复，并复跑真正的 GitHub Actions：https://github.com/tianyueni78-cyber/spinningup/actions/runs/38050751269 。对应译文 commit `fd519a9a2e065a4567702a58283c146793c54a44`，运行结果 `success`，日志有 `build succeeded`，未发现 `WARNING:` 或 `ERROR:`。
- 构建命令为 `sphinx-build -b html -n -W --keep-going`，在隔离的文档配置中执行，当前包括第 13–17 篇中文 RST。
- **明确的局限：**这份历史论文清单使用 105 条 `.. [#]` 脚注式条目，正文不以脚注编号引用它们；此前为第16篇原文的同类历史遗留现象已在隔离配置中只定向抑制 `ref.footnote` 类型警告。Sphinx 成功不表示全站依赖环境、所有旧链接仍有效，或者论文内容已做外部事实复核。

## 结论

第 17 篇：105 篇论文题录、78 条算法标注、15 条贡献说明、12 条不附注释题录以及 26 个分类标题和 107 个 URL 对照完成；译文内容审计通过；独立 Sphinx 构建通过；原站点完整集成构建仍未验收。

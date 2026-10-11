---
tags: [SpinningUp, 深度强化学习, 中文教材]
source_rst: docs/algorithms/trpo_zh.rst
source_github: https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/docs/algorithms/trpo_zh.rst
---
# 信赖域策略优化（TRPO）



## 背景

（前置内容：[VPG 背景](../algorithms/vpg_zh.md)）

TRPO 在满足新旧策略接近程度的特殊约束时，尽可能采取能够改善性能的最大更新步长。该约束使用[KL 散度](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)来表示，它衡量概率分布之间的差异，类似于距离，但严格来说并非距离。

普通策略梯度则在参数空间中保持新旧策略接近。然而，参数空间中看似微小的变化也可能带来巨大的性能差异，一次不良更新甚至可能使策略性能崩溃。因此，在原始策略梯度中使用大步长存在危险，并会损害样本效率。TRPO 可以避免这类性能崩溃，并倾向于快速、单调地提高性能。

### 基本信息

- TRPO 是同策略算法。
- TRPO 适用于离散或连续动作空间。
- Spinning Up 的 TRPO 实现支持 MPI 并行化。

### 关键公式

设 $\pi_{\theta}$ 是参数为 $\theta$ 的策略。理论上的 TRPO 更新为：

$$\begin{aligned}
\theta_{k+1} = \arg \max_{\theta} \; & {\mathcal L}(\theta_k, \theta) \\
\text{s.t.} \; & \bar{D}_{KL}(\theta || \theta_k) \leq \delta
\end{aligned}$$

其中 ${\mathcal L}(\theta_k,\theta)$ 是代理优势，它使用旧策略的数据衡量新策略 $\pi_\theta$ 相对于旧策略 $\pi_{\theta_k}$ 的表现：

$${\mathcal L}(\theta_k, \theta) = \mathbb{E}_{s,a \sim \pi_{\theta_k}}\left[
    \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a)
    \right],$$

而 $\bar{D}_{KL}(\theta || \theta_k)$ 是在旧策略访问的状态上，新旧策略之间的平均 KL 散度：

$$\bar{D}_{KL}(\theta || \theta_k) = \mathbb{E}_{s \sim \pi_{\theta_k}}\left[
    D_{KL}\left(\pi_{\theta}(\cdot|s) || \pi_{\theta_k} (\cdot|s) \right)
\right].$$

> [!note] 你应该知道
> 当 $\theta=\theta_k$ 时，目标函数和约束函数都为零。此外，约束函数对 $\theta$ 的梯度在此处也为零。证明这些结论需要细致的数学推导，准备好时很值得尝试！


理论上的 TRPO 更新并不容易直接求解，所以 TRPO 使用近似来快速得到答案。在 $\theta_k$ 附近，对目标函数和约束函数进行泰勒展开并保留最低阶项：

$$\begin{aligned}
{\mathcal L}(\theta_k, \theta) &\approx g^T (\theta - \theta_k) \\
\bar{D}_{KL}(\theta || \theta_k) & \approx \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k)
\end{aligned}$$

于是得到近似优化问题：

$$\begin{aligned}
\theta_{k+1} = \arg \max_{\theta} \; & g^T (\theta - \theta_k) \\
\text{s.t.} \; & \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k) \leq \delta.
\end{aligned}$$

> [!note] 你应该知道
> 一个巧妙的巧合是：代理优势函数对 $\theta$ 的梯度 $g$，在 $\theta=\theta_k$ 处求值时，恰好等于策略梯度 $\nabla_\theta J(\pi_\theta)$！如果愿意深入数学，可以尝试证明。


使用拉格朗日对偶方法，可以解析求解这个近似问题：

$$\theta_{k+1} = \theta_k + \sqrt{\frac{2 \delta}{g^T H^{-1} g}} H^{-1} g.$$

如果直接使用这个结果，算法实际上计算的就是[自然策略梯度](https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf)。问题在于泰勒展开引入了近似误差，更新可能违反 KL 约束，或者根本没有改善代理优势。因此，TRPO 在更新规则中增加回溯线搜索：

$$\theta_{k+1} = \theta_k + \alpha^j \sqrt{\frac{2 \delta}{g^T H^{-1} g}} H^{-1} g,$$

其中 $\alpha\in(0,1)$ 是回溯系数；$j$ 是使新策略 $\pi_{\theta_{k+1}}$ 满足 KL 约束且产生正代理优势的最小非负整数。

最后，神经网络策略可能拥有数千或数百万参数，直接计算和存储矩阵逆 $H^{-1}$ 的代价非常高。TRPO 使用[共轭梯度](https://en.wikipedia.org/wiki/Conjugate_gradient_method)算法求解 $Hx=g$，得到 $x=H^{-1}g$。它只需要计算矩阵与向量的乘积 $Hx$，不必显式计算或存储整个矩阵 $H$。我们可以构建符号运算：

$$Hx = \nabla_{\theta} \left( \left(\nabla_{\theta} \bar{D}_{KL}(\theta || \theta_k)\right)^T x \right),$$

从而在不计算整个矩阵的情况下得到正确结果。

- 文献 1： 参见 Boyd 和 Vandenberghe 的[《凸优化》](http://stanford.edu/~boyd/cvxbook/)，尤其是第 2 至第 5 章。

### 探索与利用

TRPO 以同策略方式训练随机策略，因此通过根据最新随机策略采样动作进行探索。动作选择的随机性取决于初始条件和训练过程。训练期间，更新规则鼓励利用已发现的奖励，策略通常逐渐变得不那么随机。这可能导致策略陷入局部最优。

### 伪代码


**算法伪代码：信赖域策略优化（Trust Region Policy Optimization）**

- 输入：初始策略参数 $\theta_0$，初始价值函数参数 $\phi_0$
- 超参数：KL 散度上限 $\delta$、回溯系数 $\alpha$、最大回溯步数 $K$
- **循环：** $k = 0,1,2,...$
- 在环境中运行策略 $\pi_k = \pi(\theta_k)$，收集轨迹集合 $ {\mathcal D}_k = \{\tau_i\}$。
- 计算未来回报 $\hat{R}_t$。
- 根据当前价值函数 $V_{\phi_k}$，使用任意一种优势估计方法计算优势估计 $\hat{A}_t$。
- 将策略梯度估计为
$$
\hat{g}_k = \frac{1}{|{\mathcal D}_k|} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T \left. \nabla_{\theta} \log\pi_{\theta}(a_t|s_t)\right|_{\theta_k} \hat{A}_t.
$$
- 使用共轭梯度算法计算
$$
\hat{x}_k \approx \hat{H}_k^{-1} \hat{g}_k,
$$
其中 $\hat{H}_k$ 为样本平均 KL 散度的 Hessian 矩阵。
- 使用回溯线搜索更新策略：
$$
\theta_{k+1} = \theta_k + \alpha^j \sqrt{ \frac{2\delta}{\hat{x}_k^T \hat{H}_k \hat{x}_k}} \hat{x}_k,
$$
其中 $j \in \{0, 1, 2, ... K\}$ 是使样本损失得到改善、同时满足样本 KL 散度约束的最小取值。
- 通过对均方误差进行回归来拟合价值函数：
$$
\phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T\left( V_{\phi} (s_t) - \hat{R}_t \right)^2,
$$
通常使用某种梯度下降算法完成。
- **结束循环**


## 接口文档

> [!note] 你应该知道
> Spinning Up 仅提供 TRPO 的 TensorFlow 版本。以下将原网页通过 autofunction 自动引入的函数说明译为中文。


``` python
trpo(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
     steps_per_epoch=4000, epochs=50, gamma=0.99, delta=0.01, vf_lr=1e-3,
     train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10,
     backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, logger_kwargs=dict(),
     save_freq=10, algo='trpo')
```

**信赖域策略优化。** 同时支持自然策略梯度（NPG）。

参数说明：

- env_fn：用于创建环境副本的函数，环境必须符合 OpenAI Gym API。

- actor_critic：接收状态占位符 `x_ph` 和动作占位符 `a_ph` 的函数，返回 TensorFlow 计算图的主要输出：

  <table>
  <thead>
  <tr class="header">
  <th>符号</th>
  <th>形状</th>
  <th>描述</th>
  </tr>
  </thead>
  <tbody>
  <tr class="odd">
  <td><code>pi</code></td>
  <td><blockquote>
  <p>(batch, act_dim)</p>
  </blockquote></td>
  <td><blockquote>
  <p>给定状态时，从策略采样动作。</p>
  </blockquote></td>
  </tr>
  <tr class="even">
  <td><code>logp</code></td>
  <td><blockquote>
  <p>(batch,)</p>
  </blockquote></td>
  <td><blockquote>
  <p>在状态 <code>x_ph</code> 下，策略采取动作 <code>a_ph</code> 的对数概率。</p>
  </blockquote></td>
  </tr>
  <tr class="odd">
  <td><code>logp_pi</code></td>
  <td><blockquote>
  <p>(batch,)</p>
  </blockquote></td>
  <td><blockquote>
  <p>策略对 <code>pi</code> 采样动作赋予的对数概率。</p>
  </blockquote></td>
  </tr>
  <tr class="even">
  <td><p><code>info</code></p></td>
  <td><blockquote>
  <p>N/A</p>
  </blockquote></td>
  <td><blockquote>
  <p>计算策略或对数概率时产生的中间量字典，用于解析计算</p>
  </blockquote>
  <p>KL 散度，例如概率分布的充分统计量。</p></td>
  </tr>
  <tr class="odd">
  <td><code>info_phs</code></td>
  <td><blockquote>
  <p>N/A</p>
  </blockquote></td>
  <td><blockquote>
  <p>字典，包含 <code>info</code> 各项旧值对应的占位符。</p>
  </blockquote></td>
  </tr>
  <tr class="even">
  <td><p><code>d_kl</code></p></td>
  <td><blockquote>
  <p>()</p>
  </blockquote></td>
  <td><blockquote>
  <p>用于计算当前策略 <code>pi</code> 与旧策略（由 <code>info_phs</code></p>
  </blockquote>
  <p>的输入指定）在 <code>x_ph</code> 所给状态批次上的平均 KL 散度。</p></td>
  </tr>
  <tr class="odd">
  <td><code>v</code></td>
  <td><blockquote>
  <p>(batch,)</p>
  </blockquote></td>
  <td><blockquote>
  <p><code>x_ph</code> 中状态的价值估计（必须展平为一维）。</p>
  </blockquote></td>
  </tr>
  </tbody>
  </table>

- ac_kwargs（dict）：传给 actor_critic 的其他关键字参数。

- seed（int）：随机数种子。

- steps_per_epoch（int）：每轮训练中与环境交互的步数。

- epochs（int）：训练轮数，也是策略更新次数。

- gamma（float）：折扣因子，介于 0 与 1 之间。

- delta（float）：TRPO 或 NPG 更新允许的 KL 散度上限。为了稳定性通常取 0.01 或 0.05。

- vf_lr（float）：价值函数优化器学习率。

- train_v_iters（int）：每轮训练中价值函数的梯度下降更新次数。

- damping_coeff（float）：数值稳定性阻尼系数，用于将 Hessian—向量乘积调整为：

  $$Hv \rightarrow (\alpha I+H)v$$

  其中 $\alpha$ 为阻尼系数。通常不建议调整。

- cg_iters（int）：共轭梯度迭代次数。增加迭代次数可以提高 $H^{-1}g$ 的近似精度，并可能略微改善性能，但会降低运行速度。原文还建议通常不要调整这个超参数。

- backtrack_iters（int）：回溯线搜索的最大回退次数。实际通常无需回退，或只回退一次。

- backtrack_coeff（float）：每次回溯缩小步长的比例，介于 0 和 1 之间，通常大于 0.5。

- lam（float）：GAE-Lambda 的 Lambda 参数，介于 0 和 1 之间且接近 1。

- max_ep_len（int）：一条轨迹／一个回合／一次 rollout 的最大长度。

- logger_kwargs（dict）：传递给 EpochLogger 的关键字参数。

- save_freq（int）：按训练轮次计的模型保存频率。

- algo：可以选择 trpo 或 npg；二者足够相似，因此共用这份代码。

### 保存的模型内容

日志记录器保存的 TensorFlow 计算图包括：

| 键  | 值                                      |
|-----|-----------------------------------------|
| x   | 输入状态的 TensorFlow 占位符。          |
| pi  | 根据 x 中的状态，从智能体策略采样动作。 |
| v   | 对 x 中的状态进行价值估计。             |

访问保存模型有两种方式：

- 使用 [test_policy.py](../user/saving_and_loading_zh.md) 工具运行训练后的策略；
- 使用 [restore_tf_graph](../utils/logger_zh.md) 将完整的保存计算图加载到程序中。

## 参考文献

### 相关论文

- [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477)（信赖域策略优化），Schulman 等，2015 年。
- [High Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)（使用广义优势估计进行高维连续控制），Schulman 等，2016 年。
- [Approximately Optimal Approximate Reinforcement Learning](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf)（近似最优的近似强化学习），Kakade 与 Langford，2002 年。

### 为什么选择这些论文？

Schulman 2015 是最初提出 TRPO 的论文。Schulman 2016 介绍广义优势估计，而 Spinning Up 的 TRPO 实现使用它计算策略梯度。Kakade 与 Langford 2002 提供了与 TRPO 理论基础密切相关的早期结果，也是 TRPO 的理论动机之一。

### 其他公开实现

- [Baselines](https://github.com/openai/baselines/tree/master/baselines/trpo_mpi)
- [ModularRL](https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py)
- [rllab](https://github.com/rll/rllab/blob/master/rllab/algos/trpo.py)

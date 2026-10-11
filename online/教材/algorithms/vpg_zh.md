---
tags: [SpinningUp, 深度强化学习, 中文教材]
source_rst: docs/algorithms/vpg_zh.rst
source_github: https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/docs/algorithms/vpg_zh.rst
---
# 原始策略梯度（VPG）



## 背景

（此前内容：[强化学习入门，第三部分](../spinningup/rl_intro3_zh.md)）

策略梯度背后的核心思想是：提高那些带来较高回报的动作的概率，降低那些带来较低回报的动作的概率，直到得到最优策略。

### 基本信息

- VPG 是一种同策略（on-policy）算法。
- VPG 可以用于具有离散动作空间或连续动作空间的环境。
- Spinning Up 的 VPG 实现支持利用 MPI 进行并行计算。

### 关键公式

令 $\pi_{\theta}$ 表示参数为 $\theta$ 的策略，$J(\pi_{\theta})$ 表示这个策略在有限时域、无折扣情况下的期望回报。$J(\pi_{\theta})$ 的梯度为：

$$\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}\left[
    \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) A^{\pi_{\theta}}(s_t,a_t)
    \right],$$

其中 $\tau$ 是一条轨迹，$A^{\pi_{\theta}}$ 是当前策略的优势函数。

策略梯度算法通过针对策略表现进行随机梯度上升来更新策略参数：

$$\theta_{k+1} = \theta_k + \alpha \nabla_{\theta} J(\pi_{\theta_k})$$

尽管策略梯度实现采用的是有限时域、无折扣策略梯度公式，但它们通常会基于无限时域、折扣回报来计算优势函数的估计值。

### 探索与利用

VPG 以同策略方式训练一个随机策略。这意味着，它通过按照最新版本的随机策略采样动作来进行探索。动作选择中的随机性程度取决于初始条件和训练过程。在训练期间，策略通常会逐渐变得不那么随机，因为更新规则鼓励它利用已经发现的奖励。这可能导致策略陷入局部最优。

### 伪代码


**算法伪代码：原始策略梯度算法（Vanilla Policy Gradient Algorithm）**

- 输入：初始策略参数 $\theta_0$，初始价值函数参数 $\phi_0$
- **循环：** $k = 0,1,2,...$
- 在环境中运行策略 $\pi_k = \pi(\theta_k)$，收集轨迹集合 $ {\mathcal D}_k = \{\tau_i\}$。
- 计算未来回报 $\hat{R}_t$。
- 基于当前价值函数 $V_{\phi_k}$，使用任意一种优势估计方法计算优势估计值 $\hat{A}_t$。
- 将策略梯度估计为
$$
\hat{g}_k = \frac{1}{|{\mathcal D}_k|} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T \left. \nabla_{\theta} \log\pi_{\theta}(a_t|s_t)\right|_{\theta_k} \hat{A}_t.
$$
- 计算策略更新。可以使用标准梯度上升：
$$
\theta_{k+1} = \theta_k + \alpha_k \hat{g}_k,
$$
也可以使用 Adam 等其他梯度上升算法。
- 通过最小化均方误差的回归来拟合价值函数：
$$
\phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T\left( V_{\phi} (s_t) - \hat{R}_t \right)^2,
$$
通常使用某种梯度下降算法来完成。
- **结束循环**


## 接口文档

> [!note] 你应该知道
> 以下给出 Spinning Up 中 VPG 的 PyTorch 与 TensorFlow 实现的接口文档。两种实现的函数调用与文档字符串几乎完全相同，主要区别在于模型构建细节。不过，为了内容完整，我们仍然列出两种实现的全部接口说明。


### 接口文档：PyTorch 版本

下面是 `spinup.vpg_pytorch` 的接口说明。此处以中文呈现原先由 `autofunction` 指令从源代码中自动引入的文档字符串，函数调用及参数名称保持原样。

``` python
vpg(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
    steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
    vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
    logger_kwargs=dict(), save_freq=10)
```

**原始策略梯度（Vanilla Policy Gradient）。** 使用 GAE-Lambda 估计优势。

**参数：**

`env_fn`  
创建环境副本的函数。环境必须符合 OpenAI Gym API。

`actor_critic`  
用于构造 PyTorch Module 的方法。构造得到的模块应包含 <span class="title-ref">step</span> 方法、\`act\` 方法、\`pi\` 模块和 <span class="title-ref">v</span> 模块。

<span class="title-ref">step</span> 方法应接受一批观测，并返回：

| 符号                                  | 形状             | 描述                                                                     |
|---------------------------------------|------------------|--------------------------------------------------------------------------|
| <span class="title-ref">a</span>      | (batch, act_dim) | 每个观测对应的动作所组成的 NumPy 数组。                                  |
| <span class="title-ref">v</span>      | (batch,)         | 输入观测对应的价值估计组成的 NumPy 数组。                                |
| <span class="title-ref">logp_a</span> | (batch,)         | <span class="title-ref">a</span> 中动作对应的对数概率组成的 NumPy 数组。 |

<span class="title-ref">act</span> 方法的行为与 <span class="title-ref">step</span> 相同，但只返回 <span class="title-ref">a</span>。

<span class="title-ref">pi</span> 模块的 <span class="title-ref">forward</span> 调用应接受一批观测，并可以选择接受一批动作。它应返回：

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
<td><p><span class="title-ref">pi</span></p></td>
<td><p>N/A</p></td>
<td><dl>
<dt>Torch Distribution 对象，其中包含描述</dt>
<dd>
<p>输入观测下策略的一批概率分布。</p>
</dd>
</dl></td>
</tr>
<tr class="even">
<td><p><span class="title-ref">logp_a</span></p></td>
<td><p>(batch,)</p></td>
<td><dl>
<dt>可选输出（仅在传入动作批次时返回）。</dt>
<dd>
<p>Tensor，包含策略赋予这些动作的对数概率。 未传入动作时，其值为 <span class="title-ref">None</span>。</p>
</dd>
</dl></td>
</tr>
</tbody>
</table>

<span class="title-ref">v</span> 模块的 <span class="title-ref">forward</span> 调用应接受一批观测，并返回：

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
<td><p><span class="title-ref">v</span></p></td>
<td><p>(batch,)</p></td>
<td><dl>
<dt>Tensor，包含输入观测的价值估计。</dt>
<dd>
<p>（重要：务必将其展平成一维！）</p>
</dd>
</dl></td>
</tr>
</tbody>
</table>

<span class="title-ref">ac_kwargs</span> （dict）  
传递给你为 VPG 提供的 ActorCritic 对象的任意适当关键字参数。

<span class="title-ref">seed</span> （int）  
随机数生成器的种子。

<span class="title-ref">steps_per_epoch</span> （int）  
每个训练轮次中，智能体与环境交互的步数（状态—动作对的数量）。

<span class="title-ref">epochs</span> （int）  
要执行的交互轮次数量（等同于策略更新次数）。

<span class="title-ref">gamma</span> （float）  
折扣因子。（始终位于 0 和 1 之间。）

<span class="title-ref">pi_lr</span> （float）  
策略优化器的学习率。

<span class="title-ref">vf_lr</span> （float）  
价值函数优化器的学习率。

<span class="title-ref">train_v_iters</span> （int）  
每个训练轮次中，价值函数执行梯度下降更新的步数。

<span class="title-ref">lam</span> （float）  
GAE-Lambda 中的 Lambda 参数。（始终位于 0 和 1 之间，且接近 1。）

<span class="title-ref">max_ep_len</span> （int）  
一条轨迹／一个回合／一次 rollout 的最大长度。

<span class="title-ref">logger_kwargs</span> （dict）  
提供给 EpochLogger 的关键字参数。

<span class="title-ref">save_freq</span> （int）  
保存当前策略及价值函数的频率（按训练轮次之间的间隔计算）。

### 保存的模型内容：PyTorch 版本

可以使用 <span class="title-ref">ac = torch.load('path/to/model.pt')</span> 加载保存的 PyTorch 模型，从而得到一个具有 <span class="title-ref">vpg_pytorch</span> 文档字符串所述属性的演员—评论家对象（\`ac\`）。

可以通过下面的方式，从该模型获取动作：

``` python
actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))
```

### 接口文档：TensorFlow 版本

下面是 `spinup.vpg_tf1` 的接口说明。与 PyTorch 版本一样，此处翻译的是原先通过 `autofunction` 从代码中自动生成的完整文档字符串。

``` python
vpg(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
    steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
    vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
    logger_kwargs=dict(), save_freq=10)
```

**原始策略梯度（Vanilla Policy Gradient）。** 使用 GAE-Lambda 估计优势。

**参数：**

`env_fn`  
创建环境副本的函数。环境必须满足 OpenAI Gym API 的要求。

`actor_critic`  
一个接受状态占位符符号 <span class="title-ref">x_ph</span> 和动作占位符符号 <span class="title-ref">a_ph</span> 的函数。它返回智能体 TensorFlow 计算图的主要输出：

| 符号                                   | 形状             | 描述                                                                                                              |
|----------------------------------------|------------------|-------------------------------------------------------------------------------------------------------------------|
| <span class="title-ref">pi</span>      | (batch, act_dim) | 根据状态，从策略中采样动作。                                                                                      |
| <span class="title-ref">logp</span>    | (batch,)         | 根据策略，在状态 <span class="title-ref">x_ph</span> 下采取 动作 <span class="title-ref">a_ph</span> 的对数概率。 |
| <span class="title-ref">logp_pi</span> | (batch,)         | 根据策略，由 <span class="title-ref">pi</span> 采样得到的动作 的对数概率。                                        |
| <span class="title-ref">v</span>       | (batch,)         | <span class="title-ref">x_ph</span> 中状态对应的价值估计。 （重要：务必展平成一维！）                             |

<span class="title-ref">ac_kwargs</span> （dict）  
传递给你为 VPG 提供的 <span class="title-ref">actor_critic</span> 函数的任何合适的关键字参数。

<span class="title-ref">seed</span> （int）  
随机数生成器的种子。

<span class="title-ref">steps_per_epoch</span> （int）  
每个训练轮次中，智能体与环境交互的步数（状态—动作对的数量）。

<span class="title-ref">epochs</span> （int）  
要执行的交互轮次数量（等同于策略更新次数）。

<span class="title-ref">gamma</span> （float）  
折扣因子。（始终位于 0 和 1 之间。）

<span class="title-ref">pi_lr</span> （float）  
策略优化器的学习率。

<span class="title-ref">vf_lr</span> （float）  
价值函数优化器的学习率。

<span class="title-ref">train_v_iters</span> （int）  
每个训练轮次中，价值函数执行梯度下降更新的步数。

<span class="title-ref">lam</span> （float）  
GAE-Lambda 中的 Lambda 参数。（始终位于 0 和 1 之间，且接近 1。）

<span class="title-ref">max_ep_len</span> （int）  
一条轨迹／一个回合／一次 rollout 的最大长度。

<span class="title-ref">logger_kwargs</span> （dict）  
提供给 EpochLogger 的关键字参数。

<span class="title-ref">save_freq</span> （int）  
保存当前策略及价值函数的频率（按训练轮次之间的间隔计算）。

### 保存的模型内容：TensorFlow 版本

日志记录器保存的计算图包含：

| 键                                | 值                                                                     |
|-----------------------------------|------------------------------------------------------------------------|
| <span class="title-ref">x</span>  | 用于输入状态的 TensorFlow 占位符。                                     |
| <span class="title-ref">pi</span> | 根据 <span class="title-ref">x</span> 中的状态，从智能体采样一个动作。 |
| <span class="title-ref">v</span>  | 为 <span class="title-ref">x</span> 中的状态给出价值估计。             |

可以通过以下任一方式访问保存的模型：

- 使用 [test_policy.py](../user/saving_and_loading_zh.md#loading-and-running-trained-policies) 工具运行训练后的策略；
- 或使用 [restore_tf_graph](../utils/logger_zh.md#spinup.utils.logx.restore_tf_graph) 将整个保存的计算图加载到程序中。

## 参考文献

### 相关论文

- [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)（使用函数逼近的强化学习策略梯度方法），Sutton 等，2000 年。
- [Optimizing Expectations: From Deep Reinforcement Learning to Stochastic Computation Graphs](http://joschu.net/docs/thesis.pdf)（优化期望：从深度强化学习到随机计算图），Schulman，2016(a)。
- [Benchmarking Deep Reinforcement Learning for Continuous Control](https://arxiv.org/abs/1604.06778)（面向连续控制的深度强化学习基准测试），Duan 等，2016 年。
- [High Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)（使用广义优势估计进行高维连续控制），Schulman 等，2016(b)。

### 为什么选择这些论文？

收录 Sutton 2000，是因为它是强化学习理论中历久弥新的经典之作，而且引用了促成现代策略梯度方法的更早期研究。收录 Schulman 2016(a)，是因为其第 2 章对策略梯度算法理论进行了清晰易懂的介绍，其中还包括伪代码。Duan 2016 是一篇清晰且在当时较新的基准测试论文，展示了深度强化学习环境下的原始策略梯度（例如使用神经网络策略，并以 Adam 作为优化器）与其他深度强化学习算法相比表现如何。收录 Schulman 2016(b)，则是因为我们对 VPG 的实现使用了广义优势估计来计算策略梯度。

### 其他公开实现

- [rllab](https://github.com/rll/rllab/blob/master/rllab/algos/vpg.py)
- [rllib (Ray)](https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/pg)

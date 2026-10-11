---
tags: [SpinningUp, 深度强化学习, 中文教材]
source_rst: docs/algorithms/ppo_zh.rst
source_github: https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/docs/algorithms/ppo_zh.rst
---
# 近端策略优化（PPO）



## 背景

（前文：[TRPO 的背景](../algorithms/trpo_zh.md)）

PPO 与 TRPO 源于同一个问题：如何利用当前已有的数据，尽可能大幅度地改善策略，同时又不让更新步长大到意外导致性能崩溃？TRPO 试图用复杂的二阶方法解决这一问题，而 PPO 是一类一阶方法，通过其他技巧使新策略与旧策略保持接近。PPO 的实现明显更简单，经验上似乎至少能达到与 TRPO 相当的性能。

PPO 有两种主要变体：PPO-Penalty 和 PPO-Clip。

**PPO-Penalty** 与 TRPO 类似，近似求解受 KL 散度约束的更新；但它将 KL 散度作为目标函数中的惩罚项，而不是硬约束，并在训练过程中自动调整惩罚系数，使其尺度合适。

**PPO-Clip** 的目标函数不包含 KL 散度项，也完全不使用显式约束。它改为通过对目标函数进行特殊裁剪，消除新策略远离旧策略的激励。

这里我们只讨论 PPO-Clip（OpenAI 主要采用的变体）。

### 基本信息

- PPO 是同策略（on-policy）算法。
- PPO 可用于离散动作空间和连续动作空间。
- Spinning Up 的 PPO 实现支持 MPI 并行化。

### 关键公式

PPO-Clip 通过以下方式更新策略：

$$\theta_{k+1} = \arg \max_{\theta} \underset{s,a \sim \pi_{\theta_k}}{{\mathrm E}}\left[
    L(s,a,\theta_k, \theta)\right],$$

通常通过多步随机梯度上升（一般使用小批量）最大化目标函数。其中 $L$ 定义为：

$$L(s,a,\theta_k,\theta) = \min\left(
\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\;
\text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, 1 - \epsilon, 1+\epsilon \right) A^{\pi_{\theta_k}}(s,a)
\right),$$

其中 $\epsilon$ 是一个较小的超参数，大致规定新策略可以偏离旧策略多远。

这个表达式相当复杂，乍看很难理解它的作用，以及它为什么能够让新旧策略保持接近。实际上，该目标存在一个明显简化的形式，更容易理解，也是我们在代码中实现的版本：

$$L(s,a,\theta_k,\theta) = \min\left(
\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\;
g(\epsilon, A^{\pi_{\theta_k}}(s,a))
\right),$$

其中：

$$\begin{aligned}
g(\epsilon, A) = \left\{
    \begin{array}{ll}
    (1 + \epsilon) A & A \geq 0 \ \
    (1 - \epsilon) A & A < 0.
    \end{array}
    \right.
\end{aligned}$$

为了理解这一目标，我们来看一个状态—动作对 $(s,a)$，分别考虑不同情况。

**优势为正：** 假设这个状态—动作对的优势为正，那么它对目标函数的贡献简化为：

$$L(s,a,\theta_k,\theta) = \min\left(
\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, (1 + \epsilon)
\right)  A^{\pi_{\theta_k}}(s,a).$$

因为优势为正，如果这个动作变得更可能被选择，即 $\pi_{\theta}(a|s)$ 增大，目标函数也会增大。但是 min 运算限制了目标函数能够增加的\*幅度\*。一旦 $\pi_{\theta}(a|s) > (1+\epsilon) \pi_{\theta_k}(a|s)$，min 运算便使这一项达到上限 $(1+\epsilon) A^{\pi_{\theta_k}}(s,a)$。因此，\*新策略继续远离旧策略，不会得到额外收益\*。

**优势为负：** 假设这个状态—动作对的优势为负，那么它对目标函数的贡献简化为：

$$L(s,a,\theta_k,\theta) = \max\left(
\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, (1 - \epsilon)
\right)  A^{\pi_{\theta_k}}(s,a).$$

因为优势为负，如果这个动作变得更不可能被选择，即 $\pi_{\theta}(a|s)$ 减小，目标函数便会增大。但是 max 运算限制了目标函数能够增加的\*幅度\*。一旦 $\pi_{\theta}(a|s) < (1-\epsilon) \pi_{\theta_k}(a|s)$，max 运算便使这一项达到上限 $(1-\epsilon) A^{\pi_{\theta_k}}(s,a)$。因此，同样地，\*新策略继续远离旧策略，不会得到额外收益\*。

综上，裁剪通过消除策略发生剧烈变化的激励，发挥了正则化作用；超参数 $\epsilon$ 则对应于新策略在仍能从目标函数中获益的情况下，可以偏离旧策略多远。

> [!note] 你应该知道
> 尽管裁剪能在很大程度上帮助实现合理的策略更新，新策略仍有可能偏离旧策略过远。不同 PPO 实现采用了多种技巧来避免这一问题。这里我们使用一种特别简单的方法：提前停止。当新策略相对于旧策略的平均 KL 散度超过阈值时，就停止梯度更新。
>
> 当你熟悉基本数学与实现细节之后，值得查看其他实现如何处理这一问题！


- 文献 1： 关于 PPO-Clip 目标函数简化形式的推导，参见 [这份笔记](https://drive.google.com/file/d/1PDzn9RPvaXjJFZkGeapMHbHGiWWW20Ey/view?usp=sharing)。

### 探索与利用

PPO 以同策略方式训练随机策略，因此通过从最新随机策略中采样动作进行探索。动作选择的随机性取决于初始条件和训练过程。随着训练进行，更新规则鼓励策略利用已发现的奖励，策略通常会逐渐变得不那么随机。这可能使策略陷入局部最优。

### 伪代码


**算法伪代码：PPO-Clip**

- 输入：初始策略参数 $\theta_0$，初始价值函数参数 $\phi_0$
- **循环：** $k = 0,1,2,...$
- 在环境中运行策略 $\pi_k = \pi(\theta_k)$，收集轨迹集合 ${\mathcal D}_k = \{\tau_i\}$。
- 计算未来回报 $\hat{R}_t$。
- 根据当前价值函数 $V_{\phi_k}$，用任意优势估计方法计算优势估计 $\hat{A}_t$。
- 通过最大化 PPO-Clip 目标更新策略：
$$
\theta_{k+1} = \arg \max_{\theta} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T \min\left(
\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_k}(a_t|s_t)}  A^{\pi_{\theta_k}}(s_t,a_t), \;\;
g(\epsilon, A^{\pi_{\theta_k}}(s_t,a_t))
\right),
$$
通常使用 Adam 进行随机梯度上升。
- 通过最小化均方误差拟合价值函数：
$$
\phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T\left( V_{\phi} (s_t) - \hat{R}_t \right)^2,
$$
通常使用某种梯度下降算法完成。
- **结束循环**


## 接口文档

> [!note] 你应该知道
> 下面分别给出 Spinning Up 中 PPO 的 PyTorch 和 TensorFlow 实现文档。除模型构建细节外，两者的函数调用和文档字符串几乎完全相同。为保证完整性，这里保留两套完整参数说明。


### PyTorch 版本文档

以下是原网页通过 `spinup.ppo_pytorch` 自动引入的函数说明的中文译文。

``` python
ppo(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
    steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
    vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
    target_kl=0.01, logger_kwargs=dict(), save_freq=10)
```

通过裁剪实现的近端策略优化，使用近似 KL 散度进行提前停止。

**参数**

`env_fn`  
创建环境副本的函数。环境必须符合 OpenAI Gym API。

`actor_critic`  
用于构造 PyTorch Module 的方法。该模块需要包含 `step` 方法、`act` 方法、`pi` 模块和 `v` 模块。

`step` 方法接收一批观测值，并返回：

- `a`，形状 `(batch, act_dim)`：NumPy 数组，包含每个观测值对应的动作。
- `v`，形状 `(batch,)`：NumPy 数组，包含各观测值的价值估计。
- `logp_a`，形状 `(batch,)`：NumPy 数组，包含 `a` 中各动作的对数概率。

`act` 方法与 `step` 的行为相同，但只返回 `a`。

`pi` 模块的前向调用应接收一批观测值，以及可选的一批动作，并返回：

- `pi`：Torch Distribution 对象，包含一批概率分布，描述给定观测值下的策略。
- `logp_a`，形状 `(batch,)`：可选输出。给定动作时，返回这些动作在策略下的对数概率张量；未给定动作时，其值为 `None`。

`v` 模块的前向调用应接收一批观测值并返回：

- `v`，形状 `(batch,)`：包含各观测值价值估计的张量。（非常重要：必须展平！）

`ac_kwargs` （dict）  
传递给 PPO 所使用的 ActorCritic 对象的适当关键字参数。

`seed` （int）  
随机数生成器的种子。

`steps_per_epoch` （int）  
每个训练轮次中，智能体与环境交互的步数（状态—动作对的数量）。

`epochs` （int）  
执行的交互轮次数，相当于策略更新次数。

`gamma` （float）  
折扣因子，始终位于 0 和 1 之间。

`clip_ratio` （float）  
策略目标函数中的裁剪超参数。大致表示：新策略在仍能获益（改善目标函数）的前提下，可以偏离旧策略多远？新策略仍然可能偏离得更远，但这样不会继续改善目标函数。通常取较小值，如 0.1 至 0.3，通常记作 $\epsilon$。

`pi_lr` （float）  
策略优化器的学习率。

`vf_lr` （float）  
价值函数优化器的学习率。

`train_pi_iters` （int）  
每轮训练中，对策略损失执行梯度下降的最大步数。（提前停止可能使实际步数更少。）

`train_v_iters` （int）  
每轮训练中，对价值函数执行梯度下降的步数。

`lam` （float）  
GAE-Lambda 的 Lambda 参数，始终介于 0 和 1 之间且接近 1。

`max_ep_len` （int）  
轨迹／回合／rollout 的最大长度。

`target_kl` （float）  
一次更新后，我们认为新旧策略之间适当的 KL 散度的大致数值，用于提前停止。通常较小，如 0.01 或 0.05。

`logger_kwargs` （dict）  
传递给 EpochLogger 的关键字参数。

`save_freq` （int）  
保存当前策略和价值函数的频率，以训练轮次间隔计。

### PyTorch 版本保存的模型内容

可以使用 `ac = torch.load('path/to/model.pt')` 加载 PyTorch 保存的模型，得到具有上述 `ppo_pytorch` 文档中所述属性的 actor-critic 对象 `ac`。

可通过以下代码从模型中获得动作：

``` python
actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))
```

### TensorFlow 版本文档

以下是原网页通过 `spinup.ppo_tf1` 自动引入的函数说明的中文译文。

``` python
ppo(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
    steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
    vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
    target_kl=0.01, logger_kwargs=dict(), save_freq=10)
```

通过裁剪实现的近端策略优化，使用近似 KL 散度进行提前停止。

**参数**

`env_fn`  
创建环境副本的函数。环境必须符合 OpenAI Gym API。

`actor_critic`  
接收状态占位符 `x_ph` 和动作占位符 `a_ph` 的函数，返回智能体 TensorFlow 计算图的主要输出：

- `pi`，形状 `(batch, act_dim)`：根据给定状态，从策略中采样动作。
- `logp`，形状 `(batch,)`：策略在状态 `x_ph` 下选择动作 `a_ph` 的对数概率。
- `logp_pi`，形状 `(batch,)`：策略采样得到的动作 `pi` 的对数概率。
- `v`，形状 `(batch,)`：状态 `x_ph` 的价值估计。（非常重要：必须展平！）

`ac_kwargs` （dict）  
传递给 PPO 所使用的 `actor_critic` 函数的适当关键字参数。

`seed` （int）  
随机数生成器的种子。

`steps_per_epoch` （int）  
每个训练轮次中，智能体与环境交互的步数（状态—动作对的数量）。

`epochs` （int）  
执行的交互轮次数，相当于策略更新次数。

`gamma` （float）  
折扣因子，始终位于 0 和 1 之间。

`clip_ratio` （float）  
策略目标函数中的裁剪超参数。大致表示：新策略在仍能获益（改善目标函数）的前提下，可以偏离旧策略多远？新策略仍然可能偏离得更远，但这样不会继续改善目标函数。通常取较小值，如 0.1 至 0.3，通常记作 $\epsilon$。

`pi_lr` （float）  
策略优化器的学习率。

`vf_lr` （float）  
价值函数优化器的学习率。

`train_pi_iters` （int）  
每轮训练中，对策略损失执行梯度下降的最大步数。（提前停止可能使实际步数更少。）

`train_v_iters` （int）  
每轮训练中，对价值函数执行梯度下降的步数。

`lam` （float）  
GAE-Lambda 的 Lambda 参数，始终介于 0 和 1 之间且接近 1。

`max_ep_len` （int）  
轨迹／回合／rollout 的最大长度。

`target_kl` （float）  
一次更新后，我们认为新旧策略之间适当的 KL 散度的大致数值，用于提前停止。通常较小，如 0.01 或 0.05。

`logger_kwargs` （dict）  
传递给 EpochLogger 的关键字参数。

`save_freq` （int）  
保存当前策略和价值函数的频率，以训练轮次间隔计。

### TensorFlow 版本保存的模型内容

日志记录器保存的计算图包含：

- `x`：用于状态输入的 TensorFlow 占位符。
- `pi`：根据 `x` 中的状态，从智能体采样一个动作。
- `v`：给出 `x` 中状态的价值估计。

可以通过以下两种方式访问保存的模型：

- 使用 [test_policy.py](../user/saving_and_loading_zh.md) 工具运行训练后的策略；
- 使用 [restore_tf_graph](../utils/logger_zh.md) 将保存的整个计算图加载到程序中。

## 参考文献

### 相关论文

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)，Schulman 等，2017 年。
- [High Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)，Schulman 等，2016 年。
- [Emergence of Locomotion Behaviours in Rich Environments](https://arxiv.org/abs/1707.02286)，Heess 等，2017 年。

### 为什么选择这些论文？

Schulman 2017 是最初提出 PPO 的论文。Schulman 2016 被收录，是因为我们的 PPO 实现使用广义优势估计计算策略梯度。Heess 2017 则在复杂环境中，对 PPO 智能体学到的行为进行了大规模实证分析（尽管它使用的是 PPO-Penalty，而非 PPO-Clip）。

### 其他公开实现

- [Baselines](https://github.com/openai/baselines/tree/master/baselines/ppo2)
- [ModularRL](https://github.com/joschu/modular_rl/blob/master/modular_rl/ppo.py)（注意：实现的是 PPO-Penalty，而不是 PPO-Clip。）
- [rllab](https://github.com/rll/rllab/blob/master/rllab/algos/ppo.py)（注意：实现的是 PPO-Penalty，而不是 PPO-Clip。）
- [rllib (Ray)](https://github.com/ray-project/ray/tree/master/python/ray/rllib/agents/ppo)

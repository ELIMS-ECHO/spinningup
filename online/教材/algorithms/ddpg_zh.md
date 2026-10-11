---
tags: [SpinningUp, 深度强化学习, 中文教材]
source_rst: docs/algorithms/ddpg_zh.rst
source_github: https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/docs/algorithms/ddpg_zh.rst
---
# 深度确定性策略梯度（DDPG）



## 背景

（前置内容：[强化学习导论第一部分：最优 Q 函数与最优动作](../spinningup/rl_intro_zh.md#)）

深度确定性策略梯度（DDPG）是一种同时学习 Q 函数和策略的算法。它利用离策略数据和贝尔曼方程学习 Q 函数，再利用 Q 函数学习策略。

这一方法与 Q-learning 密切相关，出发点也相同：如果已知最优动作价值函数 $Q^*(s,a)$，那么在任意给定状态下，都可以通过求解下式找到最优动作 $a^*(s)$：

$$a^*(s) = \arg \max_a Q^*(s,a).$$

DDPG 交替学习 $Q^*(s,a)$ 和 $a^*(s)$ 的近似函数，并专门针对连续动作空间设计。这里的“专门针对连续动作空间”究竟意味着什么？关键在于如何计算 $\max_a Q^*(s,a)$ 中对动作的最大化。

当离散动作的数量有限时，求最大值并不困难：只需分别计算每个动作的 Q 值并直接比较。（这样也就得到了最大化 Q 值的动作。）但在连续动作空间中，我们无法穷举所有动作，求解这一优化问题也并非易事。若使用普通优化算法，计算 $\max_a Q^*(s,a)$ 就会成为成本高昂的子程序；而智能体每次与环境交互、需要选择动作时都必须运行它，这无法接受。

由于动作空间是连续的，我们假设 $Q^*(s,a)$ 关于动作参数可微。于是可以利用这一性质，为策略 $\mu(s)$ 构造高效的、基于梯度的学习规则。这样，每当需要计算 $\max_a Q(s,a)$ 时，就不必运行昂贵的优化子程序，而可以使用近似 $\max_a Q(s,a)\approx Q(s,\mu(s))$。详见“关键公式”。

### 基本信息

- DDPG 是离策略（off-policy）算法。
- DDPG 只能用于连续动作空间环境。
- DDPG 可以理解为连续动作空间中的深度 Q-learning。
- Spinning Up 的 DDPG 实现不支持并行化。

### 关键公式

下面解释 DDPG 的两个部分：学习 Q 函数，以及学习策略。

#### DDPG 的 Q-learning 部分

首先回顾描述最优动作价值函数 $Q^*(s,a)$ 的贝尔曼方程：

$$Q^*(s,a) = \underset{s' \sim P}{{\mathrm E}}\left[r(s,a) + \gamma \max_{a'} Q^*(s', a')\right]$$

其中 $s'\sim P$ 是简写，表示环境从分布 $P(\cdot|s,a)$ 中采样下一状态 $s'$。

这个贝尔曼方程是学习 $Q^*(s,a)$ 近似函数的起点。假设近似函数是参数为 $\phi$ 的神经网络 $Q_\phi(s,a)$，并且已经收集了转移数据集 $\mathcal D$，其中每个转移为 $(s,a,r,s',d)$，$d$ 表示 $s'$ 是否为终止状态。可以定义\**均方贝尔曼误差（MSBE）*\*，大致衡量 $Q_\phi$ 满足贝尔曼方程的程度：

$$L(\phi, {\mathcal D}) = \underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[
    \Bigg( Q_{\phi}(s,a) - \left(r + \gamma (1 - d) \max_{a'} Q_{\phi}(s',a') \right) \Bigg)^2
    \right]$$

这里计算 $(1-d)$ 时，使用了 Python 的约定：True 视为 1，False 视为 0。因此当 d==True，即 $s'$ 为终止状态时，Q 函数应反映智能体在当前转移之后不会获得额外奖励。（这一记号也与后面的代码实现一致。）

使用函数近似的 Q-learning 算法，包括 DQN 及其各种变体和 DDPG，很大程度上都基于最小化这一 MSBE 损失。它们共同使用两项值得说明的技巧；随后再介绍 DDPG 特有的细节。

**技巧一：经验回放缓冲区。** 所有使用深度神经网络近似 $Q^*(s,a)$ 的标准算法都会使用经验回放缓冲区，即保存先前经验的集合 $\mathcal D$。为了使算法运行稳定，缓冲区应足够大，容纳多种经验，但保留所有历史数据也未必总是有益。如果只使用最新的数据，就会对其过拟合，导致训练出问题；如果使用过多经验，又可能降低学习速度。实际使用时可能需要调参。

> [!note] 你应该知道
> 我们提到 DDPG 是离策略算法，现在正好解释为什么以及如何做到这一点。注意，回放缓冲区\*应该\*包含旧经验，即使这些经验由已经过时的策略产生。为什么仍然能够使用它们？因为贝尔曼方程\*不关心\*使用哪些转移元组、动作如何选择，也不关心某次转移之后发生什么。最优 Q 函数应当对\*所有\*可能的转移都满足贝尔曼方程。因此，曾经经历过的任何转移，都可以用于通过最小化 MSBE 拟合 Q 函数近似器。


**技巧二：目标网络。** Q-learning 算法使用\**目标网络*\*。下式：

$$r + \gamma (1 - d) \max_{a'} Q_{\phi}(s',a')$$

称为\**目标值（target）*\*，因为最小化 MSBE 时，我们希望 Q 函数的输出接近这个目标值。问题在于目标值也依赖正在训练的同一组参数 $\phi$，这使 MSBE 的最小化过程不稳定。解决办法是使用一组与 $\phi$ 接近但更新滞后的参数，也就是第二个网络：目标网络。其参数记作 $\phi_{\text{targ}}$。

在基于 DQN 的算法中，目标网络每隔固定步数直接复制主网络的参数。在 DDPG 一类算法中，目标网络在主网络每次更新后，使用 Polyak 平均进行一次更新：

$$\phi_{\text{targ}} \leftarrow \rho \phi_{\text{targ}} + (1 - \rho) \phi,$$

其中 $\rho$ 是介于 0 与 1 之间、通常接近 1 的超参数。（代码中称为 polyak。）

**DDPG 特有的细节：计算目标值中对动作的最大化。** 如前所述，在连续动作空间中，计算目标值里的动作最大值十分困难。DDPG 使用\**目标策略网络*\*计算一个近似最大化 $Q_{\phi_{\text{targ}}}$ 的动作。目标策略网络的获得方式与目标 Q 函数相同：在训练过程中对策略参数进行 Polyak 平均。

综合上述内容，DDPG 的 Q-learning 部分通过随机梯度下降，最小化以下 MSBE 损失：

$$L(\phi, {\mathcal D}) = \underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[
    \Bigg( Q_{\phi}(s,a) - \left(r + \gamma (1 - d) Q_{\phi_{\text{targ}}}(s', \mu_{\theta_{\text{targ}}}(s')) \right) \Bigg)^2
    \right],$$

其中 $\mu_{\theta_{\text{targ}}}$ 是目标策略。

#### DDPG 的策略学习部分

DDPG 的策略学习相当简单。我们希望学习一个确定性策略 $\mu_\theta(s)$，使其给出最大化 $Q_\phi(s,a)$ 的动作。由于动作空间连续，并且假设 Q 函数关于动作可微，因此只需针对策略参数执行梯度上升，求解：

$$\max_{\theta} \underset{s \sim {\mathcal D}}{{\mathrm E}}\left[ Q_{\phi}(s, \mu_{\theta}(s)) \right].$$

注意，这里将 Q 函数的参数视为常量。

### 探索与利用

DDPG 以离策略方式训练确定性策略。由于策略是确定性的，如果智能体以同策略方式探索，在训练初期很可能无法尝试足够多样的动作，从而难以发现有用的学习信号。为了提高 DDPG 的探索能力，训练时我们会向策略输出的动作添加噪声。原始 DDPG 论文作者推荐具有时间相关性的 [OU 噪声](https://en.wikipedia.org/wiki/Ornstein%E2%80%93Uhlenbeck_process)，但较新的研究结果表明，彼此不相关、均值为零的高斯噪声同样表现良好。由于后者更简单，因此更受青睐。为了获得质量更高的训练数据，可以在训练过程中逐渐降低噪声幅度。（我们的实现没有这么做，而是始终保持固定噪声幅度。）

在测试时，为了检验策略利用已学知识的能力，我们不会向动作添加噪声。

> [!note] 你应该知道
> 我们的 DDPG 实现使用一个技巧来改善训练初期的探索。在最开始固定数量的步数内（由 start_steps 关键字参数设定），智能体从合法动作范围内的均匀随机分布中采样动作。此后才恢复正常的 DDPG 探索。


### 伪代码


**算法伪代码：深度确定性策略梯度**

- 输入：初始策略参数 $\theta$、Q 函数参数 $\phi$、空经验回放缓冲区 $\mathcal D$
- 将目标参数设为主网络参数：$\theta_{\text{targ}}\leftarrow\theta$，$\phi_{\text{targ}}\leftarrow\phi$
- **重复：**
- 观察状态 $s$，选择动作 $a=\text{clip}(\mu_\theta(s)+\epsilon,a_{Low},a_{High})$，其中 $\epsilon\sim\mathcal N$
- 在环境中执行动作 $a$
- 观察下一状态 $s'$、奖励 $r$ 和表示 $s'$ 是否为终止状态的完成信号 $d$
- 将 $(s,a,r,s',d)$ 存入回放缓冲区 $\mathcal D$
- 如果 $s'$ 是终止状态，则重置环境状态
- **如果：** 到达更新时机
- **循环：** 执行预定次数的更新
- 从 $\mathcal D$ 中随机采样一批转移 $B=\{(s,a,r,s',d)\}$
- 计算目标值
$$
y(r,s',d)=r+\gamma(1-d)Q_{\phi_{\text{targ}}}(s',\mu_{\theta_{\text{targ}}}(s'))
$$
- 使用下式进行一步梯度下降，更新 Q 函数
$$
\nabla_\phi\frac{1}{|B|}\sum_{(s,a,r,s',d)\in B}
\left(Q_\phi(s,a)-y(r,s',d)\right)^2
$$
- 使用下式进行一步梯度上升，更新策略
$$
\nabla_\theta\frac{1}{|B|}\sum_{s\in B}Q_\phi(s,\mu_\theta(s))
$$
- 更新目标网络：
$$
\begin{aligned}
\phi_{\text{targ}} &\leftarrow \rho\phi_{\text{targ}}+(1-\rho)\phi\\
\theta_{\text{targ}} &\leftarrow \rho\theta_{\text{targ}}+(1-\rho)\theta
\end{aligned}
$$
- **结束循环**
- **结束条件**
- **直到：** 收敛


## 接口文档

> [!note] 你应该知道
> 以下分别提供 Spinning Up 中 DDPG 的 PyTorch 与 TensorFlow 实现文档。除了模型构造相关细节，两者的函数调用和文档字符串几乎相同。为求完整，下面仍分别列出两套说明。


### PyTorch 版本

原始页面使用 `autofunction` 指令引入 `spinup.ddpg_pytorch` 的源码文档。这里将其展开翻译。

``` python
ddpg(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
     steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
     polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
     update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10,
     max_ep_len=1000, logger_kwargs=dict(), save_freq=1)
```

**深度确定性策略梯度（DDPG）**

**参数：**

env_fn  
创建环境副本的函数。环境必须符合 OpenAI Gym API。

actor_critic  
PyTorch Module 的构造函数。该模块必须包含 act 方法、pi 模块和 q 模块。act 方法与 pi 模块应接收一批观测值，q 模块应接收一批观测值及一批动作。调用结果应为：

- act：形状为 (batch, act_dim) 的 NumPy 数组，包含每个观测对应的动作。
- pi：形状为 (batch, act_dim) 的 Tensor，包含策略根据观测生成的动作。
- q：形状为 (batch,) 的 Tensor，包含给定观测和动作对应的最优动作价值函数 $Q^*$ 的当前估计。（非常重要：务必将其展平！）

ac_kwargs（dict）  
适用于提供给 DDPG 的 ActorCritic 对象的任意关键字参数。

seed（int）  
随机数生成器种子。

steps_per_epoch（int）  
每轮中智能体与环境交互的步数，即状态—动作对数量。

epochs（int）  
智能体运行和训练的轮数。

replay_size（int）  
经验回放缓冲区的最大长度。

gamma（float）  
折扣因子，始终位于 0 和 1 之间。

polyak（float）  
目标网络 Polyak 平均的插值系数。目标网络向主网络靠近的更新规则为：

$$\theta_{\text{targ}}\leftarrow\rho\theta_{\text{targ}}+(1-\rho)\theta$$

其中 $\rho$ 即 polyak，始终介于 0 和 1 之间，通常接近 1。

pi_lr（float）  
策略的学习率。

q_lr（float）  
Q 网络的学习率。

batch_size（int）  
随机梯度下降所用的小批量大小。

start_steps（int）  
开始使用真实策略之前，采取均匀随机动作的步数，有助于探索。

update_after（int）  
开始执行梯度下降更新前，需要收集的环境交互次数，以确保回放缓冲区有足够数据支持有效更新。

update_every（int）  
两次梯度下降更新之间经过的环境交互次数。注意，无论两次更新之间等待多久，环境步数与梯度更新步数之比固定为 1。

act_noise（float）  
训练时添加到策略动作上的高斯探索噪声标准差。（测试时不添加噪声。）

num_test_episodes（int）  
每轮结束时测试确定性策略的回合数。

max_ep_len（int）  
轨迹／回合／rollout 的最大长度。

logger_kwargs（dict）  
传递给 EpochLogger 的关键字参数。

save_freq（int）  
保存当前策略和价值函数的频率，以训练轮次之间的间隔计。

### PyTorch 版本保存的模型

可以使用以下代码加载 PyTorch 保存的模型：

``` python
ac = torch.load('path/to/model.pt')
```

这将得到一个 actor-critic 对象 `ac`，其属性如 `ddpg_pytorch` 的函数文档所述。

可以通过以下代码让模型产生动作：

``` python
actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))
```

### TensorFlow 版本

原始页面使用 `autofunction` 指令引入 `spinup.ddpg_tf1` 的源码文档。以下完整翻译其参数说明。

``` python
ddpg(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
     steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
     polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
     update_after=1000, update_every=50, act_noise=0.1, num_test_episodes=10,
     max_ep_len=1000, logger_kwargs=dict(), save_freq=1)
```

**深度确定性策略梯度（DDPG）**

**参数：**

env_fn  
创建环境副本的函数。环境必须符合 OpenAI Gym API。

actor_critic  
一个接收状态占位符 x_ph 与动作占位符 a_ph 的函数，返回智能体 TensorFlow 计算图的主要输出：

- pi：形状 (batch, act_dim)，根据状态确定性地计算策略动作。
- q：形状 (batch,)，对 x_ph 中的状态及 a_ph 中的动作给出最优动作价值函数 $Q^*$ 的当前估计。
- q_pi：形状 (batch,)，在 x_ph 中的状态上组合 q 与 pi，即 q(x, pi(x))。

ac_kwargs（dict）  
适用于提供给 DDPG 的 actor_critic 函数的任意关键字参数。

seed（int）  
随机数生成器种子。

steps_per_epoch（int）  
每轮智能体与环境交互的步数（状态—动作对数量）。

epochs（int）  
智能体运行与训练的轮数。

replay_size（int）  
经验回放缓冲区的最大长度。

gamma（float）  
折扣因子，始终介于 0 和 1 之间。

polyak（float）  
目标网络进行 Polyak 平均时的插值系数。目标网络参数按照以下公式向主网络参数更新：

$$\theta_{\text{targ}}\leftarrow\rho\theta_{\text{targ}}+(1-\rho)\theta$$

其中 $\rho$ 为 polyak，始终介于 0 和 1 之间，通常接近 1。

pi_lr（float）  
策略的学习率。

q_lr（float）  
Q 网络的学习率。

batch_size（int）  
随机梯度下降所用的小批量大小。

start_steps（int）  
使用真实策略之前，先执行均匀随机动作的步数，有助于探索。

update_after（int）  
开始执行梯度下降更新之前需要收集的环境交互次数，确保回放缓冲区具有足够数据。

update_every（int）  
梯度下降更新之间间隔的环境交互次数。注意：无论等待间隔有多长，环境步数与梯度更新步数之比固定为 1。

act_noise（float）  
训练时添加到策略动作上的高斯探索噪声的标准差。（测试时不添加噪声。）

num_test_episodes（int）  
每轮结束时用于测试确定性策略的回合数。

max_ep_len（int）  
轨迹／回合／rollout 的最大长度。

logger_kwargs（dict）  
传给 EpochLogger 的关键字参数。

save_freq（int）  
保存当前策略和价值函数的频率，以轮次间隔计。

### TensorFlow 版本保存的模型

日志记录器保存的计算图包含：

- x：用于输入状态的 TensorFlow 占位符。
- a：用于输入动作的 TensorFlow 占位符。
- pi：根据 x 中的状态，确定性地计算智能体动作。
- q：给出 x 中状态与 a 中动作对应的动作价值估计。

访问保存模型有两种方式：

- 使用 [test_policy.py](../user/saving_and_loading_zh.md#) 工具运行训练后的策略。
- 使用 [restore_tf_graph](../utils/logger_zh.md#) 将保存的完整计算图加载到程序中。

## 参考文献

### 相关论文

- [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf)（确定性策略梯度算法），Silver 等，2014 年。
- [Continuous Control With Deep Reinforcement Learning](https://arxiv.org/abs/1509.02971)（利用深度强化学习进行连续控制），Lillicrap 等，2016 年。

### 为什么选择这些论文？

收录 Silver 2014，是因为它建立了确定性策略梯度（DPG）的基础理论。收录 Lillicrap 2016，是因为它将具有理论基础的 DPG 算法扩展到深度强化学习场景，从而提出 DDPG。

### 其他公开实现

- [Baselines](https://github.com/openai/baselines/tree/master/baselines/ddpg)
- [rllab](https://github.com/rll/rllab/blob/master/rllab/algos/ddpg.py)
- [rllib (Ray)](https://github.com/ray-project/ray/tree/master/python/ray/rllib/agents/ddpg)
- [TD3 发布仓库](https://github.com/sfujim/TD3)

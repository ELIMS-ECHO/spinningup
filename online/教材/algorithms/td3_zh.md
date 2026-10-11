---
tags: [SpinningUp, 深度强化学习, 中文教材]
source_rst: docs/algorithms/td3_zh.rst
source_github: https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/docs/algorithms/td3_zh.rst
---
# 双延迟 DDPG（TD3）



## 背景

（前文：[DDPG 的背景](../algorithms/ddpg_zh.md#background)）

虽然 DDPG 有时能取得出色表现，但它经常对超参数和其他调节方式十分敏感。DDPG 的一种常见失败模式是：学到的 Q 函数开始严重高估 Q 值，随后策略利用 Q 函数中的误差，最终导致策略失效。双延迟 DDPG（TD3）通过引入三项关键技巧解决这一问题：

**技巧一：裁剪双 Q 学习。** TD3 学习\*两个\* Q 函数，而非一个（因此称为“双”）；在构造贝尔曼误差损失的目标值时，取两个 Q 值中较小的一个。

**技巧二：“延迟”策略更新。** TD3 更新策略（及目标网络）的频率低于更新 Q 函数的频率。论文建议每更新两次 Q 函数，才更新一次策略。

**技巧三：目标策略平滑。** TD3 对目标动作添加噪声，使 Q 函数随动作变化更加平滑，从而降低策略利用 Q 函数误差的可能性。

这三项技巧共同作用，使 TD3 的表现比基础 DDPG 有显著改善。

### 基本信息

- TD3 是离策略算法。
- TD3 只能用于连续动作空间环境。
- Spinning Up 的 TD3 实现不支持并行化。

### 关键公式

TD3 通过最小化均方贝尔曼误差，同时学习两个 Q 函数 $Q_{\phi_1}$ 和 $Q_{\phi_2}$，方式与 DDPG 学习单个 Q 函数基本相同。为了准确展示 TD3 的做法及其与普通 DDPG 的差别，我们从损失函数最内部的部分开始，逐步向外解释。

首先是\**目标策略平滑*\*。用于构造 Q-learning 目标值的动作以目标策略 $\mu_{\theta_{\text{targ}}}$ 为基础，但会在动作的每个维度添加经过裁剪的噪声。添加噪声后，目标动作还要再次裁剪，使其处于合法动作范围内（所有合法动作 $a$ 都满足 $a_{Low}\leq a\leq a_{High}$）。因此目标动作为：

$$a'(s') = \text{clip}\left(\mu_{\theta_{\text{targ}}}(s') + \text{clip}(\epsilon,-c,c), a_{Low}, a_{High}\right), \;\;\;\;\; \epsilon \sim \mathcal{N}(0, \sigma)$$

目标策略平滑实际上是算法的一种正则化手段。它针对 DDPG 可能出现的一种特定失败模式：如果 Q 函数近似器错误地为某些动作产生尖锐峰值，策略就会迅速利用这个峰值，进而表现得脆弱或错误。对相近动作上的 Q 函数进行平滑可以避免这一问题，而目标策略平滑正是为此设计的。

其次是\**裁剪双 Q 学习*\*。两个 Q 函数使用同一个目标值，该目标值通过两个目标 Q 函数中较小的估计值计算：

$$y(r,s',d) = r + \gamma (1 - d) \min_{i=1,2} Q_{\phi_{i, \text{targ}}}(s', a'(s')),$$

随后两个 Q 函数都通过回归逼近该目标值来学习：

$$L(\phi_1, {\mathcal D}) = \mathbb{E}_{(s,a,r,s',d) \sim {\mathcal D}}\left[
    \Bigg( Q_{\phi_1}(s,a) - y(r,s',d) \Bigg)^2
    \right],$$

$$L(\phi_2, {\mathcal D}) = \mathbb{E}_{(s,a,r,s',d) \sim {\mathcal D}}\left[
    \Bigg( Q_{\phi_2}(s,a) - y(r,s',d) \Bigg)^2
    \right].$$

使用较小的 Q 值构造目标，并向这个目标回归，有助于抑制 Q 函数的过高估计。

最后，策略仅通过最大化 $Q_{\phi_1}$ 来学习：

$$\max_{\theta} \underset{s \sim {\mathcal D}}{{\mathrm E}}\left[ Q_{\phi_1}(s, \mu_{\theta}(s)) \right],$$

这一点与 DDPG 基本相同。不过在 TD3 中，策略的更新频率低于 Q 函数。这有助于缓解 DDPG 中常见的波动，因为策略更新会改变目标值。

### 探索与利用

TD3 以离策略方式训练确定性策略。由于策略是确定性的，如果智能体在训练初期按照策略本身进行探索，就可能无法尝试足够多样的动作，也就难以找到有用的学习信号。为了增强 TD3 策略的探索能力，训练时我们向动作添加噪声，通常使用互不相关、均值为零的高斯噪声。为了获得质量更高的训练数据，也可以在训练过程中逐渐减小噪声幅度。（我们的实现没有这样做，而是始终保持固定噪声幅度。）

测试时，为了考察策略利用所学知识的能力，我们不会向动作添加噪声。

> [!note] 你应该知道
> 我们的 TD3 实现采用一项技巧来改善训练初期的探索：在开始的固定步数内（通过 `start_steps` 关键字参数设置），智能体从合法动作范围内的均匀随机分布采样动作。之后才恢复正常的 TD3 探索。


### 伪代码


**算法伪代码：双延迟 DDPG**

- 输入：初始策略参数 $\theta$、Q 函数参数 $\phi_1$、$\phi_2$，以及空回放缓冲区 $\mathcal D$
- 令目标参数等于主网络参数：$\theta_{\text{targ}}\leftarrow\theta$、$\phi_{\text{targ},1}\leftarrow\phi_1$、$\phi_{\text{targ},2}\leftarrow\phi_2$
- **重复：**
- 观察状态 $s$ 并选择动作 $a=\text{clip}(\mu_\theta(s)+\epsilon,a_{Low},a_{High})$，其中 $\epsilon\sim\mathcal N$
- 在环境中执行 $a$
- 观察下一状态 $s'$、奖励 $r$ 和表示 $s'$ 是否为终止状态的完成信号 $d$
- 将 $(s,a,r,s',d)$ 存入回放缓冲区 $\mathcal D$
- 如果 $s'$ 是终止状态，则重置环境状态
- **如果：** 到了更新时机
- **循环：** $j$ 取预定的更新次数
- 从 $\mathcal D$ 中随机采样一批转移 $B=\{(s,a,r,s',d)\}$
- 计算目标动作
$
a'(s')=\text{clip}\left(\mu_{\theta_{\text{targ}}}(s')+\text{clip}(\epsilon,-c,c),a_{Low},a_{High}\right),\quad \epsilon\sim\mathcal N(0,\sigma)
$
- 计算目标值
$
y(r,s',d)=r+\gamma(1-d)\min_{i=1,2}Q_{\phi_{\text{targ},i}}(s',a'(s'))
$
- 利用下式执行一步梯度下降，更新两个 Q 函数
\begin{align*}
&\nabla_{\phi_i}\frac{1}{|B|}\sum_{(s,a,r,s',d)\in B}\left(Q_{\phi_i}(s,a)-y(r,s',d)\right)^2 &&\text{对于 }i=1,2
\end{align*}
- **如果：** $j\mod\texttt{policy\_delay}=0$
- 利用下式执行一步梯度上升，更新策略
$
\nabla_\theta\frac{1}{|B|}\sum_{s\in B}Q_{\phi_1}(s,\mu_\theta(s))
$
- 更新目标网络
\begin{align*}
\phi_{\text{targ},i}&\leftarrow\rho\phi_{\text{targ},i}+(1-\rho)\phi_i &&\text{对于 }i=1,2\\
\theta_{\text{targ}}&\leftarrow\rho\theta_{\text{targ}}+(1-\rho)\theta
\end{align*}
- **结束条件**
- **结束循环**
- **结束条件**
- **直到：** 收敛


## 接口文档

> [!note] 你应该知道
> 以下分别给出 Spinning Up 中 TD3 的 PyTorch 和 TensorFlow 实现文档。除模型构造方面的细节外，两者的函数调用和文档字符串几乎相同。为保证完整性，这里分别列出两套函数说明。


### PyTorch 版本接口

原网页通过 `autofunction` 指令引用 `spinup.td3_pytorch` 的函数文档；以下将源码文档展开翻译。

``` python
td3(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
    steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
    polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
    update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
    noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
    logger_kwargs=dict(), save_freq=1)
```

**双延迟深度确定性策略梯度（TD3）**

**参数：**

env_fn  
创建环境副本的函数，环境必须符合 OpenAI Gym API。

actor_critic  
用于构造 PyTorch Module 的方法。该模块必须具有 act 方法、pi 模块、q1 模块及 q2 模块。act 和 pi 接收一批观测作为输入；q1 与 q2 接收一批观测和一批动作作为输入。各自返回：

- act：形状 (batch, act_dim)，针对每个观测返回动作的 NumPy 数组。
- pi：形状 (batch, act_dim)，包含策略针对观测生成动作的 Tensor。
- q1：形状 (batch,)，包含给定观测与动作对应的最优动作价值函数 $Q^*$ 的第一组当前估计。（重要：务必展平！）
- q2：形状 (batch,)，包含给定观测与动作对应的最优动作价值函数 $Q^*$ 的另一组当前估计。（重要：务必展平！）

ac_kwargs（dict）  
适用于提供给 TD3 的 ActorCritic 对象的任意关键字参数。

seed（int）  
随机数生成器种子。

steps_per_epoch（int）  
每轮智能体与环境交互的步数（状态—动作对数量）。

epochs（int）  
运行并训练智能体的轮数。

replay_size（int）  
经验回放缓冲区的最大长度。

gamma（float）  
折扣因子，始终介于 0 和 1 之间。

polyak（float）  
目标网络 Polyak 平均的插值系数。目标网络按以下方式向主网络更新：

$$\theta_{\text{targ}}\leftarrow\rho\theta_{\text{targ}}+(1-\rho)\theta$$

其中 $\rho$ 为 polyak，始终介于 0 和 1 之间，通常接近 1。

pi_lr（float）  
策略的学习率。

q_lr（float）  
Q 网络的学习率。

batch_size（int）  
随机梯度下降的小批量大小。

start_steps（int）  
在运行真正策略之前，先采用均匀随机动作的步数，有助于探索。

update_after（int）  
开始梯度下降更新之前需要收集的环境交互次数，确保回放缓冲区拥有足够数据。

update_every（int）  
梯度下降更新之间的环境交互次数。注意：无论两次更新之间等待多久，环境步数与梯度更新步数之比都固定为 1。

act_noise（float）  
训练时添加到策略动作上的高斯探索噪声标准差。（测试时不添加噪声。）

target_noise（float）  
添加到目标策略的平滑噪声标准差。

noise_clip（float）  
目标策略平滑噪声绝对值的上限。

policy_delay（int）  
每更新 Q 网络 policy_delay 次，策略才更新一次。

num_test_episodes（int）  
每轮结束时用于测试确定性策略的回合数。

max_ep_len（int）  
轨迹／回合／rollout 的最大长度。

logger_kwargs（dict）  
传给 EpochLogger 的关键字参数。

save_freq（int）  
保存当前策略和价值函数的频率，以训练轮次之间的间隔计。

### PyTorch 版本保存的模型

PyTorch 保存的模型可以通过以下方式加载：

``` python
ac = torch.load('path/to/model.pt')
```

这将得到一个 actor-critic 对象 `ac`，其属性与 `td3_pytorch` 函数文档中的描述一致。

可以通过以下方式从模型获取动作：

``` python
actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))
```

### TensorFlow 版本接口

原网页通过 `autofunction` 指令引用 `spinup.td3_tf1` 的函数文档；以下为源码文档的完整中文参数说明。

``` python
td3(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
    steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
    polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000,
    update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
    noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
    logger_kwargs=dict(), save_freq=1)
```

**双延迟深度确定性策略梯度（TD3）**

**参数：**

env_fn  
创建环境副本的函数。环境必须符合 OpenAI Gym API。

actor_critic  
接收状态占位符 x_ph 和动作占位符 a_ph 的函数，返回智能体 TensorFlow 计算图的主要输出：

- pi：形状 (batch, act_dim)，根据给定状态确定性地计算策略动作。
- q1：形状 (batch,)，针对 x_ph 中的状态和 a_ph 中的动作给出对最优动作价值函数 $Q^*$ 的一组估计。
- q2：形状 (batch,)，针对 x_ph 中的状态和 a_ph 中的动作给出另对最优动作价值函数 $Q^*$ 的一组估计。
- q1_pi：形状 (batch,)，给出 x_ph 中状态下 q1 与 pi 的组合，即 q1(x, pi(x))。

ac_kwargs（dict）  
适用于提供给 TD3 的 actor_critic 函数的任意关键字参数。

seed（int）  
随机数生成器种子。

steps_per_epoch（int）  
每轮智能体与环境交互的步数，即状态—动作对数量。

epochs（int）  
运行并训练智能体的轮数。

replay_size（int）  
经验回放缓冲区的最大长度。

gamma（float）  
折扣因子，始终介于 0 和 1 之间。

polyak（float）  
目标网络 Polyak 平均的插值系数。目标网络向主网络更新的方式为：

$$\theta_{\text{targ}}\leftarrow\rho\theta_{\text{targ}}+(1-\rho)\theta$$

其中 $\rho$ 为 polyak，始终介于 0 和 1 之间，通常接近 1。

pi_lr（float）  
策略的学习率。

q_lr（float）  
Q 网络的学习率。

batch_size（int）  
随机梯度下降的小批量大小。

start_steps（int）  
在运行真正策略之前，先采用均匀随机动作的步数，有助于探索。

update_after（int）  
开始梯度下降更新之前需要收集的环境交互次数，确保回放缓冲区拥有足够数据。

update_every（int）  
梯度下降更新之间的环境交互次数。注意：无论两次更新之间等待多久，环境步数与梯度更新步数之比都固定为 1。

act_noise（float）  
训练时添加到策略动作上的高斯探索噪声标准差。（测试时不添加噪声。）

target_noise（float）  
添加到目标策略的平滑噪声标准差。

noise_clip（float）  
目标策略平滑噪声绝对值的上限。

policy_delay（int）  
每更新 Q 网络 policy_delay 次，策略才更新一次。

num_test_episodes（int）  
每轮结束时用于测试确定性策略的回合数。

max_ep_len（int）  
轨迹／回合／rollout 的最大长度。

logger_kwargs（dict）  
传给 EpochLogger 的关键字参数。

save_freq（int）  
保存当前策略和价值函数的频率，以训练轮次之间的间隔计。

### TensorFlow 版本保存的模型

日志记录器保存的计算图包括：

- x：用于状态输入的 TensorFlow 占位符。
- a：用于动作输入的 TensorFlow 占位符。
- pi：根据 x 中的状态，确定性地计算智能体动作。
- q1：给出 x 中状态与 a 中动作对应的一组动作价值估计。
- q2：给出 x 中状态与 a 中动作对应的另一组动作价值估计。

可以通过以下两种方式访问保存的模型：

- 使用 [test_policy.py](../user/saving_and_loading_zh.md#loading-and-running-trained-policies) 工具运行训练后的策略；
- 使用 [restore_tf_graph](../utils/logger_zh.md#spinup.utils.logx.restore_tf_graph) 将完整的保存计算图加载到程序中。

## 参考文献

### 相关论文

- [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477)（解决 Actor-Critic 方法中的函数近似误差），Fujimoto 等，2018 年。

### 其他公开实现

- [TD3 发布仓库](https://github.com/sfujim/TD3)

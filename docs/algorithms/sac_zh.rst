==========================
软演员—评论家算法（SAC）
==========================

.. contents:: 目录

背景
==========

（前文：`TD3 的背景`_）

.. _`TD3 的背景`: ../algorithms/td3.html#background

软演员—评论家（Soft Actor-Critic，SAC）是一种以离策略方式优化随机策略的算法，连接了随机策略优化方法和 DDPG 一类方法。它并不是 TD3 的直接后继算法（两者几乎同时发表），但它采用了裁剪双 Q 技巧，而且由于 SAC 策略本身具有随机性，也能获得类似于目标策略平滑的效果。

SAC 的核心特征是**熵正则化**。策略训练的目标是在期望回报与`熵`_之间寻求平衡，熵衡量策略的随机程度。这与探索和利用之间的权衡密切相关：提高熵会增加探索，可能加快后续学习，也有助于防止策略过早收敛到较差的局部最优。

.. _`熵`: https://en.wikipedia.org/wiki/Entropy_(information_theory)

基本信息
-----------

* SAC 是离策略算法。
* 这里实现的 SAC 版本只能用于连续动作空间。
* 也可以实现另一种 SAC 版本，稍微修改策略更新规则即可处理离散动作空间。
* Spinning Up 的 SAC 实现不支持并行化。

关键公式
-------------

为了解释 SAC，首先需要介绍熵正则化强化学习。在熵正则化强化学习中，价值函数的定义会稍有不同。

熵正则化强化学习
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

熵大致描述一个随机变量的随机程度。如果一枚硬币被调整得几乎总是正面朝上，它的熵就很低；如果硬币均匀，正反两面各有一半概率，它的熵就很高。

设 :math:`x` 为随机变量，概率质量函数或概率密度函数为 :math:`P`。根据其分布 :math:`P`，熵 :math:`H` 计算如下：

.. math::

    H(P) = \underE{x \sim P}{-\log P(x)}.

在熵正则化强化学习中，智能体在每个时间步都会获得一个与该时间步策略熵成比例的奖励加成。于是`强化学习问题`_ 变为：

.. math::

    \pi^* = \arg \max_{\pi} \underE{\tau \sim \pi}{ \sum_{t=0}^{\infty} \gamma^t \bigg( R(s_t, a_t, s_{t+1}) + \alpha H\left(\pi(\cdot|s_t)\right) \bigg)},

其中 :math:`\alpha>0` 为权衡系数。（注意：这里假设无限时域折扣设置，本文后续部分也采用这一假设。）

现在可以定义该设置下稍有不同的价值函数。:math:`V^\pi` 被修改为包含每个时间步的熵奖励：

.. math::

    V^{\pi}(s) = \underE{\tau \sim \pi}{ \left. \sum_{t=0}^{\infty} \gamma^t \bigg( R(s_t, a_t, s_{t+1}) + \alpha H\left(\pi(\cdot|s_t)\right) \bigg) \right| s_0 = s}

:math:`Q^\pi` 则被修改为包含*除第一个时间步之外*所有时间步的熵奖励：

.. math::

    Q^{\pi}(s,a) = \underE{\tau \sim \pi}{ \left. \sum_{t=0}^{\infty} \gamma^t R(s_t, a_t, s_{t+1}) + \alpha \sum_{t=1}^{\infty} \gamma^t H\left(\pi(\cdot|s_t)\right)\right| s_0 = s, a_0 = a}

根据这些定义，:math:`V^\pi` 与 :math:`Q^\pi` 的关系为：

.. math::

    V^{\pi}(s) = \underE{a \sim \pi}{Q^{\pi}(s,a)} + \alpha H\left(\pi(\cdot|s)\right)

而 :math:`Q^\pi` 的贝尔曼方程为：

.. math::

    Q^{\pi}(s,a) &= \underE{s' \sim P \\ a' \sim \pi}{R(s,a,s') + \gamma\left(Q^{\pi}(s',a') + \alpha H\left(\pi(\cdot|s')\right) \right)} \\
    &= \underE{s' \sim P}{R(s,a,s') + \gamma V^{\pi}(s')}.

.. _`强化学习问题`: ../spinningup/rl_intro.html#the-rl-problem

.. admonition:: 你应该知道

    这里对熵正则化价值函数的定义带有一定任意性。实际上，我们完全可以采用其他定义，例如让 :math:`Q^\pi` 也包含第一个时间步的熵奖励。不同论文在这一主题上的定义可能略有差异。

软演员—评论家算法
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SAC 同时学习策略 :math:`\pi_\theta` 和两个 Q 函数 :math:`Q_{\phi_1},Q_{\phi_2}`。目前标准的 SAC 主要有两种变体：一种使用固定的熵正则化系数 :math:`\alpha`；另一种通过训练期间改变 :math:`\alpha` 来满足熵约束。为简单起见，Spinning Up 使用固定熵正则化系数的版本，但实践者通常更偏好带熵约束的变体。

.. admonition:: 你应该知道

    SAC 算法随时间有所变化。较旧版本除 Q 函数外还学习价值函数 :math:`V_\psi`；本文主要讨论省略这个额外价值函数的现代版本。

**学习 Q 函数。** Q 函数的学习与 TD3 类似，但存在几项关键差异。

首先看相同之处：

1. 和 TD3 一样，两个 Q 函数都通过最小化 MSBE、向同一个共享目标回归来学习。
2. 和 TD3 一样，共享目标使用目标 Q 网络计算；目标 Q 网络通过训练期间对 Q 网络参数进行 Polyak 平均获得。
3. 和 TD3 一样，共享目标采用**裁剪双 Q** 技巧。

不同之处是：

1. 与 TD3 不同，目标值还包含 SAC 熵正则化带来的额外项。
2. 与 TD3 不同，目标值中的下一状态动作来自**当前策略**，而不是目标策略。
3. 与 TD3 不同，SAC 没有显式的目标策略平滑。TD3 训练确定性策略，因此通过向下一状态动作添加随机噪声实现平滑；SAC 训练随机策略，其自身随机性产生的噪声已足以达到类似效果。

在给出最终的 Q 损失函数之前，先解释熵正则化项如何进入目标值。回到前面熵正则化 :math:`Q^\pi` 的递归贝尔曼方程，利用熵的定义进行改写：

.. math::

    Q^{\pi}(s,a) &= \underE{s' \sim P \\ a' \sim \pi}{R(s,a,s') + \gamma\left(Q^{\pi}(s',a') + \alpha H\left(\pi(\cdot|s')\right) \right)} \\
    &= \underE{s' \sim P \\ a' \sim \pi}{R(s,a,s') + \gamma\left(Q^{\pi}(s',a') - \alpha \log \pi(a'|s') \right)}

右侧是对下一状态和下一动作求期望。下一状态来自经验回放缓冲区，下一动作来自当前策略，**不是**回放缓冲区。既然是期望，就可以通过采样近似：

.. math::

    Q^{\pi}(s,a) &\approx r + \gamma\left(Q^{\pi}(s',\tilde{a}') - \alpha \log \pi(\tilde{a}'|s') \right), \;\;\;\;\; \tilde{a}' \sim \pi(\cdot|s').

.. admonition:: 你应该知道

    这里将下一动作记号从 :math:`a'` 改为 :math:`\tilde a'`，是为了强调下一动作必须重新从策略采样；相比之下，:math:`r` 和 :math:`s'` 应来自经验回放缓冲区。

SAC 使用这种对目标值的采样近似，为每个 Q 函数构造 MSBE 损失。唯一尚未确定的是，采样备份时使用哪个 Q 函数。与 TD3 一样，SAC 采用裁剪双 Q 技巧，取两个 Q 近似器估计值中的较小者。

综上，SAC 的 Q 网络损失为：

.. math::

    L(\phi_i, {\mathcal D}) = \underset{(s,a,r,s',d) \sim {\mathcal D}}{{\mathrm E}}\left[
        \Bigg( Q_{\phi_i}(s,a) - y(r,s',d) \Bigg)^2
        \right],

其中目标值为：

.. math::

    y(r, s', d) = r + \gamma (1 - d) \left( \min_{j=1,2} Q_{\phi_{\text{targ},j}}(s', \tilde{a}') - \alpha \log \pi_{\theta}(\tilde{a}'|s') \right), \;\;\;\;\; \tilde{a}' \sim \pi_{\theta}(\cdot|s').

**学习策略。** 策略在每个状态下都应该使预期未来回报与预期未来熵之和最大，即最大化 :math:`V^\pi(s)`。展开可得：

.. math::

    V^{\pi}(s) &= \underE{a \sim \pi}{Q^{\pi}(s,a)} + \alpha H\left(\pi(\cdot|s)\right) \\
    &= \underE{a \sim \pi}{Q^{\pi}(s,a) - \alpha \log \pi(a|s)}.

我们使用**重参数化技巧**优化策略：通过计算状态、策略参数和独立噪声的确定性函数，得到服从 :math:`\pi_\theta(\cdot|s)` 的样本。举例来说，遵循 SAC 论文作者的做法，我们使用经过压缩变换的高斯策略（squashed Gaussian policy），样本按下式获得：

.. math::

    \tilde{a}_{\theta}(s, \xi) = \tanh\left( \mu_{\theta}(s) + \sigma_{\theta}(s) \odot \xi \right), \;\;\;\;\; \xi \sim \mathcal{N}(0, I).

.. admonition:: 你应该知道

    这种策略与其他策略优化算法中的策略存在两项关键区别：

    **1. 压缩函数。** SAC 策略中的 :math:`\tanh` 确保动作限制在有限范围内。VPG、TRPO 和 PPO 的策略没有这个操作。它还会改变动作分布：在经过 :math:`\tanh` 之前，SAC 策略与其他算法一样是分解的高斯分布；经过 :math:`\tanh` 之后则不再是高斯分布。（不过仍可以解析计算动作的对数概率，详见论文附录。）

    **2. 标准差的参数化方式。** 在 VPG、TRPO 和 PPO 中，我们使用与状态无关的参数向量表示对数标准差；而在 SAC 中，对数标准差是神经网络的输出，因此以复杂方式依赖于状态。根据我们的经验，采用与状态无关的对数标准差的 SAC 无法正常工作。（你能想出原因吗？或者更进一步，设计实验验证？）

重参数化技巧允许我们把对动作的期望（困难在于动作分布依赖策略参数）改写为对噪声的期望（困难由此消除，因为噪声分布不依赖参数）：

.. math::

    \underE{a \sim \pi_{\theta}}{Q^{\pi_{\theta}}(s,a) - \alpha \log \pi_{\theta}(a|s)} = \underE{\xi \sim \mathcal{N}}{Q^{\pi_{\theta}}(s,\tilde{a}_{\theta}(s,\xi)) - \alpha \log \pi_{\theta}(\tilde{a}_{\theta}(s,\xi)|s)}

为了得到策略损失，最后还需要使用函数近似器替换 :math:`Q^{\pi_\theta}`。与 TD3 使用第一个 Q 近似器 :math:`Q_{\phi_1}` 不同，SAC 使用两个 Q 近似器的最小值 :math:`\min_{j=1,2}Q_{\phi_j}`。因此策略优化目标为：

.. math::

    \max_{\theta} \underE{s \sim \mathcal{D} \\ \xi \sim \mathcal{N}}{\min_{j=1,2} Q_{\phi_j}(s,\tilde{a}_{\theta}(s,\xi)) - \alpha \log \pi_{\theta}(\tilde{a}_{\theta}(s,\xi)|s)},

除了取双 Q 最小值、随机性和熵项之外，这与 DDPG、TD3 的策略优化几乎相同。

探索与利用
----------------------------

SAC 训练带有熵正则化的随机策略，并以同策略方式进行探索。熵正则化系数 :math:`\alpha` 显式控制探索与利用之间的权衡：:math:`\alpha` 越大，探索越多；:math:`\alpha` 越小，利用越多。合适的系数（能够实现最稳定学习或最高奖励的系数）可能因环境而异，需要仔细调整。

测试时，为了观察策略利用所学知识的能力，我们移除随机性，使用均值动作，而不是从分布中采样动作。这样通常比原始随机策略表现更好。

.. admonition:: 你应该知道

    我们的 SAC 实现使用一个技巧来改善训练初期的探索：在最开始固定数量的步数内（通过 start_steps 关键字参数设置），智能体从合法动作范围的均匀随机分布中采样动作。之后才恢复正常 SAC 探索。

伪代码
----------

.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{软演员—评论家算法}
        \label{alg1}
    \begin{algorithmic}[1]
        \STATE 输入：初始策略参数 $\theta$、Q 函数参数 $\phi_1$ 和 $\phi_2$，以及空经验回放缓冲区 $\mathcal D$
        \STATE 将目标参数设为主网络参数：$\phi_{\text{targ},1}\leftarrow\phi_1$，$\phi_{\text{targ},2}\leftarrow\phi_2$
        \REPEAT
            \STATE 观察状态 $s$，选择动作 $a\sim\pi_\theta(\cdot|s)$
            \STATE 在环境中执行 $a$
            \STATE 观察下一状态 $s'$、奖励 $r$，以及表示 $s'$ 是否为终止状态的完成信号 $d$
            \STATE 将 $(s,a,r,s',d)$ 存入经验回放缓冲区 $\mathcal D$
            \STATE 如果 $s'$ 是终止状态，则重置环境状态
            \IF{到了更新时机}
                \FOR{$j$ 遍历预定的更新次数}
                    \STATE 从 $\mathcal D$ 中随机采样一批转移 $B=\{(s,a,r,s',d)\}$
                    \STATE 计算 Q 函数的目标值：
                    \begin{align*}
                        y(r,s',d)&=r+\gamma(1-d)\left(\min_{i=1,2}Q_{\phi_{\text{targ},i}}(s',\tilde a')-\alpha\log\pi_\theta(\tilde a'|s')\right), &&\tilde a'\sim\pi_\theta(\cdot|s')
                    \end{align*}
                    \STATE 使用下式进行一步梯度下降，更新两个 Q 函数：
                    \begin{align*}
                        &\nabla_{\phi_i}\frac{1}{|B|}\sum_{(s,a,r,s',d)\in B}\left(Q_{\phi_i}(s,a)-y(r,s',d)\right)^2 &&\text{对于 }i=1,2
                    \end{align*}
                    \STATE 使用下式进行一步梯度上升，更新策略：
                    \begin{equation*}
                        \nabla_\theta\frac{1}{|B|}\sum_{s\in B}\Big(\min_{i=1,2}Q_{\phi_i}(s,\tilde a_\theta(s))-\alpha\log\pi_\theta\left(\left.\tilde a_\theta(s)\right|s\right)\Big),
                    \end{equation*}
                    其中 $\tilde a_\theta(s)$ 是从 $\pi_\theta(\cdot|s)$ 采样的动作，并可通过重参数化技巧对 $\theta$ 求导。
                    \STATE 更新目标网络：
                    \begin{align*}
                        \phi_{\text{targ},i}&\leftarrow\rho\phi_{\text{targ},i}+(1-\rho)\phi_i &&\text{对于 }i=1,2
                    \end{align*}
                \ENDFOR
            \ENDIF
        \UNTIL{收敛}
    \end{algorithmic}
    \end{algorithm}

接口文档
==========

.. admonition:: 你应该知道

    以下提供 Spinning Up 中 SAC 的 PyTorch 和 TensorFlow 实现文档。除了模型构造相关细节，两者的函数调用和文档字符串几乎相同。为求完整，这里分别给出两套文档字符串的中文翻译。

PyTorch 版本接口
------------------------------

原始网页使用 autofunction 指令引用 spinup.sac_pytorch；以下展开翻译源码中的函数文档。

.. code-block:: python

    sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=1)

**软演员—评论家算法（SAC）**

**参数：**

env_fn
    创建环境副本的函数。环境必须符合 OpenAI Gym API。

actor_critic
    PyTorch Module 的构造方法。该模块应具有 act 方法、pi 模块、q1 模块和 q2 模块。act 方法和 pi 模块接收一批观测作为输入，q1 与 q2 接收一批观测和一批动作作为输入。各方法应返回：

    * act：形状 (batch, act_dim) 的 NumPy 数组，包含每个观测对应的动作。
    * q1：形状 (batch,) 的 Tensor，包含给定观测和动作对应的最优 Q 值的第一组当前估计。（务必展平！）
    * q2：形状 (batch,) 的 Tensor，包含给定观测和动作对应的最优 Q 值的另一组当前估计。（务必展平！）

    调用 pi 应返回：

    * a：形状 (batch, act_dim) 的 Tensor，包含策略根据观测生成的动作。
    * logp_pi：形状 (batch,) 的 Tensor，包含 a 中动作的对数概率。重要的是：梯度必须能够反向传播到 a。

ac_kwargs（dict）
    适用于提供给 SAC 的 ActorCritic 对象的任意关键字参数。

seed（int）
    随机数生成器的种子。

steps_per_epoch（int）
    每轮智能体与环境交互的步数，即状态—动作对的数量。

epochs（int）
    智能体运行与训练的轮数。

replay_size（int）
    经验回放缓冲区的最大长度。

gamma（float）
    折扣因子，始终介于 0 和 1 之间。

polyak（float）
    目标网络 Polyak 平均的插值系数。目标网络向主网络更新的规则为：

    .. math::

        \theta_{\text{targ}}\leftarrow\rho\theta_{\text{targ}}+(1-\rho)\theta

    其中 :math:`\rho` 为 polyak，始终介于 0 和 1 之间，通常接近 1。

lr（float）
    学习率，同时用于策略学习和价值学习。

alpha（float）
    熵正则化系数。（等价于原始 SAC 论文中奖励尺度的倒数。）

batch_size（int）
    随机梯度下降的小批量大小。

start_steps（int）
    在运行真实策略之前，先采取均匀随机动作的步数，有助于探索。

update_after（int）
    开始梯度下降更新前需要收集的环境交互次数，确保经验回放缓冲区中已有足够数据用于有效更新。

update_every（int）
    梯度下降更新之间经过的环境交互次数。注意，无论两次更新之间等待多久，环境步数与梯度更新步数之比固定为 1。

num_test_episodes（int）
    每轮结束时测试确定性策略的回合数。

max_ep_len（int）
    轨迹／回合／rollout 的最大长度。

logger_kwargs（dict）
    传给 EpochLogger 的关键字参数。

save_freq（int）
    保存当前策略和价值函数的频率，以训练轮次之间的间隔计。

PyTorch 版本保存的模型
-------------------------------------

可以使用以下代码加载 PyTorch 保存的模型：

.. code-block:: python

    ac = torch.load('path/to/model.pt')

得到的 actor-critic 对象 ac 具有 sac_pytorch 函数文档中描述的属性。

可通过以下代码从模型获取动作：

.. code-block:: python

    actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))

TensorFlow 版本接口
------------------------------

原始网页使用 autofunction 指令引用 spinup.sac_tf1；以下展开翻译源码中的函数文档。

.. code-block:: python

    sac(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000,
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=1)

**软演员—评论家算法（SAC）**

**参数：**

env_fn
    创建环境副本的函数。环境必须符合 OpenAI Gym API。

actor_critic
    接收状态占位符 x_ph 和动作占位符 a_ph 的函数，返回智能体 TensorFlow 计算图的主要输出：

    * mu：形状 (batch, act_dim)，根据状态计算策略的均值动作。
    * pi：形状 (batch, act_dim)，根据状态从策略中采样动作。
    * logp_pi：形状 (batch,)，给出策略对 pi 所采样动作的对数概率。关键要求：从动作采样过程一直到策略参数，必须全程可微。
    * q1：形状 (batch,)，给出 x_ph 中状态和 a_ph 中动作对应的一组最优 Q 值估计。
    * q2：形状 (batch,)，给出 x_ph 中状态和 a_ph 中动作对应的另一组最优 Q 值估计。

ac_kwargs（dict）
    适用于提供给 SAC 的 actor_critic 函数的任意关键字参数。

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
    目标网络 Polyak 平均的插值系数。目标网络按以下规则向主网络更新：

    .. math::

        \theta_{\text{targ}}\leftarrow\rho\theta_{\text{targ}}+(1-\rho)\theta

    其中 :math:`\rho` 为 polyak，始终介于 0 和 1 之间，通常接近 1。

lr（float）
    学习率，同时用于策略学习与价值学习。

alpha（float）
    熵正则化系数。（等价于原始 SAC 论文中奖励尺度的倒数。）

batch_size（int）
    随机梯度下降的小批量大小。

start_steps（int）
    开始运行真实策略之前，采取均匀随机动作的步数，有助于探索。

update_after（int）
    开始梯度下降更新前需要收集的环境交互次数，确保回放缓冲区中有足够数据。

update_every（int）
    梯度下降更新之间经过的环境交互次数。注意，无论两次更新之间等待多久，环境步数与梯度更新步数之比固定为 1。

num_test_episodes（int）
    每轮结束时测试确定性策略的回合数。

max_ep_len（int）
    轨迹／回合／rollout 的最大长度。

logger_kwargs（dict）
    传给 EpochLogger 的关键字参数。

save_freq（int）
    保存当前策略和价值函数的频率，以训练轮次之间的间隔计。

TensorFlow 版本保存的模型
----------------------------------------

日志记录器保存的计算图包括：

* x：用于状态输入的 TensorFlow 占位符。
* a：用于动作输入的 TensorFlow 占位符。
* mu：根据 x 中的状态，确定性地计算智能体的均值动作。
* pi：根据 x 中的状态，从智能体策略中采样动作。
* q1：给出 x 中状态与 a 中动作对应的一组动作价值估计。
* q2：给出 x 中状态与 a 中动作对应的另一组动作价值估计。
* v：给出 x 中状态的价值估计。

访问保存模型有两种方式：

* 使用 test_policy.py_ 工具运行训练后的策略；
* 使用 restore_tf_graph_ 将完整保存的计算图加载到程序中。

注意：对于 SAC，正确的评估策略由 mu 给出，而不是 pi。可以把 pi 看作探索策略，把 mu 看作利用策略。

.. _test_policy.py: ../user/saving_and_loading.html#loading-and-running-trained-policies
.. _restore_tf_graph: ../utils/logger.html#spinup.utils.logx.restore_tf_graph

参考文献
==========

相关论文
---------------

* `Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor`_（软演员—评论家：使用随机 Actor 的离策略最大熵深度强化学习），Haarnoja 等，2018 年。
* `Soft Actor-Critic Algorithms and Applications`_（软演员—评论家算法及其应用），Haarnoja 等，2018 年。
* `Learning to Walk via Deep Reinforcement Learning`_（通过深度强化学习学习行走），Haarnoja 等，2018 年。

.. _`Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor`: https://arxiv.org/abs/1801.01290
.. _`Soft Actor-Critic Algorithms and Applications`: https://arxiv.org/abs/1812.05905
.. _`Learning to Walk via Deep Reinforcement Learning`: https://arxiv.org/abs/1812.11103

其他公开实现
----------------------------

* `SAC 发布仓库`_（最初的“官方”代码库）
* `Softlearning 仓库`_（目前的“官方”代码库）
* `Yarats 与 Kostrikov 仓库`_

.. _`SAC 发布仓库`: https://github.com/haarnoja/sac
.. _`Softlearning 仓库`: https://github.com/rail-berkeley/softlearning
.. _`Yarats 与 Kostrikov 仓库`: https://github.com/denisyarats/pytorch_sac

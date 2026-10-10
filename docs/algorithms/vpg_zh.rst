========================
原始策略梯度（VPG）
========================

.. contents:: 目录


背景
==========

（此前内容：`强化学习入门，第三部分`_）

.. _`强化学习入门，第三部分`: ../spinningup/rl_intro3.html

策略梯度背后的核心思想是：提高那些带来较高回报的动作的概率，降低那些带来较低回报的动作的概率，直到得到最优策略。


基本信息
-----------

* VPG 是一种同策略（on-policy）算法。
* VPG 可以用于具有离散动作空间或连续动作空间的环境。
* Spinning Up 的 VPG 实现支持利用 MPI 进行并行计算。


关键公式
-------------

令 :math:`\pi_{\theta}` 表示参数为 :math:`\theta` 的策略，:math:`J(\pi_{\theta})` 表示这个策略在有限时域、无折扣情况下的期望回报。:math:`J(\pi_{\theta})` 的梯度为：

.. math::

    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{
        \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) A^{\pi_{\theta}}(s_t,a_t)
        },

其中 :math:`\tau` 是一条轨迹，:math:`A^{\pi_{\theta}}` 是当前策略的优势函数。

策略梯度算法通过针对策略表现进行随机梯度上升来更新策略参数：

.. math::

    \theta_{k+1} = \theta_k + \alpha \nabla_{\theta} J(\pi_{\theta_k})

尽管策略梯度实现采用的是有限时域、无折扣策略梯度公式，但它们通常会基于无限时域、折扣回报来计算优势函数的估计值。


探索与利用
----------------------------

VPG 以同策略方式训练一个随机策略。这意味着，它通过按照最新版本的随机策略采样动作来进行探索。动作选择中的随机性程度取决于初始条件和训练过程。在训练期间，策略通常会逐渐变得不那么随机，因为更新规则鼓励它利用已经发现的奖励。这可能导致策略陷入局部最优。


伪代码
----------

.. math::
    :nowrap:

    \begin{algorithm}[H]
        \caption{原始策略梯度算法（Vanilla Policy Gradient Algorithm）}
        \label{alg1}
    \begin{algorithmic}[1]
        \STATE 输入：初始策略参数 $\theta_0$，初始价值函数参数 $\phi_0$
        \FOR{$k = 0,1,2,...$}
        \STATE 在环境中运行策略 $\pi_k = \pi(\theta_k)$，收集轨迹集合 $ {\mathcal D}_k = \{\tau_i\}$。
        \STATE 计算未来回报 $\hat{R}_t$。
        \STATE 基于当前价值函数 $V_{\phi_k}$，使用任意一种优势估计方法计算优势估计值 $\hat{A}_t$。
        \STATE 将策略梯度估计为
            \begin{equation*}
            \hat{g}_k = \frac{1}{|{\mathcal D}_k|} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T \left. \nabla_{\theta} \log\pi_{\theta}(a_t|s_t)\right|_{\theta_k} \hat{A}_t.
            \end{equation*}
        \STATE 计算策略更新。可以使用标准梯度上升：
            \begin{equation*}
            \theta_{k+1} = \theta_k + \alpha_k \hat{g}_k,
            \end{equation*}
            也可以使用 Adam 等其他梯度上升算法。
        \STATE 通过最小化均方误差的回归来拟合价值函数：
            \begin{equation*}
            \phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T\left( V_{\phi} (s_t) - \hat{R}_t \right)^2,
            \end{equation*}
            通常使用某种梯度下降算法来完成。
        \ENDFOR
    \end{algorithmic}
    \end{algorithm}


接口文档
=============

.. admonition:: 你应该知道

    以下给出 Spinning Up 中 VPG 的 PyTorch 与 TensorFlow 实现的接口文档。两种实现的函数调用与文档字符串几乎完全相同，主要区别在于模型构建细节。不过，为了内容完整，我们仍然列出两种实现的全部接口说明。


接口文档：PyTorch 版本
------------------------------

下面是 `spinup.vpg_pytorch` 的接口说明。此处以中文呈现原先由 `autofunction` 指令从源代码中自动引入的文档字符串，函数调用及参数名称保持原样。

.. code-block:: python

    vpg(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, pi_lr=3e-4,
        vf_lr=1e-3, train_v_iters=80, lam=0.97, max_ep_len=1000,
        logger_kwargs=dict(), save_freq=10)

**原始策略梯度（Vanilla Policy Gradient）。** 使用 GAE-Lambda 估计优势。

**参数：**

`env_fn`
    创建环境副本的函数。环境必须符合 OpenAI Gym API。

`actor_critic`
    用于构造 PyTorch Module 的方法。构造得到的模块应包含 `step` 方法、`act` 方法、`pi` 模块和 `v` 模块。

    `step` 方法应接受一批观测，并返回：

    ===========  ================  ============================================
    符号         形状              描述
    ===========  ================  ============================================
    `a`          (batch, act_dim)  每个观测对应的动作所组成的 NumPy 数组。
    `v`          (batch,)          输入观测对应的价值估计组成的 NumPy 数组。
    `logp_a`     (batch,)          `a` 中动作对应的对数概率组成的 NumPy 数组。
    ===========  ================  ============================================

    `act` 方法的行为与 `step` 相同，但只返回 `a`。

    `pi` 模块的 `forward` 调用应接受一批观测，并可以选择接受一批动作。它应返回：

    ===========  ================  ============================================
    符号         形状              描述
    ===========  ================  ============================================
    `pi`         N/A               Torch Distribution 对象，其中包含描述
                                    输入观测下策略的一批概率分布。
    `logp_a`     (batch,)          可选输出（仅在传入动作批次时返回）。
                                    Tensor，包含策略赋予这些动作的对数概率。
                                    未传入动作时，其值为 `None`。
    ===========  ================  ============================================

    `v` 模块的 `forward` 调用应接受一批观测，并返回：

    ===========  ================  ============================================
    符号         形状              描述
    ===========  ================  ============================================
    `v`          (batch,)          Tensor，包含输入观测的价值估计。
                                    （重要：务必将其展平成一维！）
    ===========  ================  ============================================

`ac_kwargs` （dict）
    传递给你为 VPG 提供的 ActorCritic 对象的任意适当关键字参数。

`seed` （int）
    随机数生成器的种子。

`steps_per_epoch` （int）
    每个训练轮次中，智能体与环境交互的步数（状态—动作对的数量）。

`epochs` （int）
    要执行的交互轮次数量（等同于策略更新次数）。

`gamma` （float）
    折扣因子。（始终位于 0 和 1 之间。）

`pi_lr` （float）
    策略优化器的学习率。

`vf_lr` （float）
    价值函数优化器的学习率。

`train_v_iters` （int）
    每个训练轮次中，价值函数执行梯度下降更新的步数。

`lam` （float）
    GAE-Lambda 中的 Lambda 参数。（始终位于 0 和 1 之间，且接近 1。）

`max_ep_len` （int）
    一条轨迹／一个回合／一次 rollout 的最大长度。

`logger_kwargs` （dict）
    提供给 EpochLogger 的关键字参数。

`save_freq` （int）
    保存当前策略及价值函数的频率（按训练轮次之间的间隔计算）。

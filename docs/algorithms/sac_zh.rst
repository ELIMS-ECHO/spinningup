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

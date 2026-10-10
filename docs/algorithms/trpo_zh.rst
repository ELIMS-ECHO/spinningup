================================
信赖域策略优化（TRPO）
================================

.. contents:: 目录

背景
==========

（前置内容：`VPG 背景`_）

.. _`VPG 背景`: ../algorithms/vpg.html#background

TRPO 在满足新旧策略接近程度的特殊约束时，尽可能采取能够改善性能的最大更新步长。该约束使用`KL 散度`_来表示，它衡量概率分布之间的差异，类似于距离，但严格来说并非距离。

普通策略梯度则在参数空间中保持新旧策略接近。然而，参数空间中看似微小的变化也可能带来巨大的性能差异，一次不良更新甚至可能使策略性能崩溃。因此，在原始策略梯度中使用大步长存在危险，并会损害样本效率。TRPO 可以避免这类性能崩溃，并倾向于快速、单调地提高性能。

.. _`KL 散度`: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

基本信息
-----------

* TRPO 是同策略算法。
* TRPO 适用于离散或连续动作空间。
* Spinning Up 的 TRPO 实现支持 MPI 并行化。

关键公式
-------------

设 :math:`\pi_{\theta}` 是参数为 :math:`\theta` 的策略。理论上的 TRPO 更新为：

.. math::

    \theta_{k+1} = \arg \max_{\theta} \; & {\mathcal L}(\theta_k, \theta) \\
    \text{s.t.} \; & \bar{D}_{KL}(\theta || \theta_k) \leq \delta

其中 :math:`{\mathcal L}(\theta_k,\theta)` 是代理优势，它使用旧策略的数据衡量新策略 :math:`\pi_\theta` 相对于旧策略 :math:`\pi_{\theta_k}` 的表现：

.. math::

    {\mathcal L}(\theta_k, \theta) = \underE{s,a \sim \pi_{\theta_k}}{
        \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)} A^{\pi_{\theta_k}}(s,a)
        },

而 :math:`\bar{D}_{KL}(\theta || \theta_k)` 是在旧策略访问的状态上，新旧策略之间的平均 KL 散度：

.. math::

    \bar{D}_{KL}(\theta || \theta_k) = \underE{s \sim \pi_{\theta_k}}{
        D_{KL}\left(\pi_{\theta}(\cdot|s) || \pi_{\theta_k} (\cdot|s) \right)
    }.

.. admonition:: 你应该知道

    当 :math:`\theta=\theta_k` 时，目标函数和约束函数都为零。此外，约束函数对 :math:`\theta` 的梯度在此处也为零。证明这些结论需要细致的数学推导，准备好时很值得尝试！

理论上的 TRPO 更新并不容易直接求解，所以 TRPO 使用近似来快速得到答案。在 :math:`\theta_k` 附近，对目标函数和约束函数进行泰勒展开并保留最低阶项：

.. math::

    {\mathcal L}(\theta_k, \theta) &\approx g^T (\theta - \theta_k) \\
    \bar{D}_{KL}(\theta || \theta_k) & \approx \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k)

于是得到近似优化问题：

.. math::

    \theta_{k+1} = \arg \max_{\theta} \; & g^T (\theta - \theta_k) \\
    \text{s.t.} \; & \frac{1}{2} (\theta - \theta_k)^T H (\theta - \theta_k) \leq \delta.

.. admonition:: 你应该知道

    一个巧妙的巧合是：代理优势函数对 :math:`\theta` 的梯度 :math:`g`，在 :math:`\theta=\theta_k` 处求值时，恰好等于策略梯度 :math:`\nabla_\theta J(\pi_\theta)`！如果愿意深入数学，可以尝试证明。

使用拉格朗日对偶方法 [1]_，可以解析求解这个近似问题：

.. math::

    \theta_{k+1} = \theta_k + \sqrt{\frac{2 \delta}{g^T H^{-1} g}} H^{-1} g.

如果直接使用这个结果，算法实际上计算的就是`自然策略梯度`_。问题在于泰勒展开引入了近似误差，更新可能违反 KL 约束，或者根本没有改善代理优势。因此，TRPO 在更新规则中增加回溯线搜索：

.. math::

    \theta_{k+1} = \theta_k + \alpha^j \sqrt{\frac{2 \delta}{g^T H^{-1} g}} H^{-1} g,

其中 :math:`\alpha\in(0,1)` 是回溯系数；:math:`j` 是使新策略 :math:`\pi_{\theta_{k+1}}` 满足 KL 约束且产生正代理优势的最小非负整数。

最后，神经网络策略可能拥有数千或数百万参数，直接计算和存储矩阵逆 :math:`H^{-1}` 的代价非常高。TRPO 使用`共轭梯度`_算法求解 :math:`Hx=g`，得到 :math:`x=H^{-1}g`。它只需要计算矩阵与向量的乘积 :math:`Hx`，不必显式计算或存储整个矩阵 :math:`H`。我们可以构建符号运算：

.. math::

    Hx = \nabla_{\theta} \left( \left(\nabla_{\theta} \bar{D}_{KL}(\theta || \theta_k)\right)^T x \right),

从而在不计算整个矩阵的情况下得到正确结果。

.. [1] 参见 Boyd 和 Vandenberghe 的`《凸优化》`_，尤其是第 2 至第 5 章。

.. _`《凸优化》`: http://stanford.edu/~boyd/cvxbook/
.. _`自然策略梯度`: https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf
.. _`共轭梯度`: https://en.wikipedia.org/wiki/Conjugate_gradient_method

探索与利用
----------------------------

TRPO 以同策略方式训练随机策略，因此通过根据最新随机策略采样动作进行探索。动作选择的随机性取决于初始条件和训练过程。训练期间，更新规则鼓励利用已发现的奖励，策略通常逐渐变得不那么随机。这可能导致策略陷入局部最优。

伪代码
----------

.. math::
    :nowrap:

    \begin{algorithm}[H]
    \caption{信赖域策略优化算法}
    \begin{algorithmic}[1]
    \STATE 输入初始策略参数 $\theta_0$ 和价值函数参数 $\phi_0$。
    \FOR{$k=0,1,2,\ldots$}
    \STATE 运行策略 $\pi_k=\pi(\theta_k)$，收集轨迹集合 $\mathcal D_k$。
    \STATE 计算未来回报 $\hat R_t$ 与优势估计 $\hat A_t$。
    \STATE 计算策略梯度估计：
    \begin{equation*}
    \hat g_k = \frac{1}{|\mathcal D_k|}\sum_{\tau\in\mathcal D_k}\sum_{t=0}^T
    \left.\nabla_\theta\log\pi_\theta(a_t|s_t)\right|_{\theta_k}\hat A_t.
    \end{equation*}
    \STATE 用共轭梯度法求解 $\hat x_k\approx \hat H_k^{-1}\hat g_k$。
    \STATE 回溯线搜索，取满足 KL 约束并改善代理目标的最小 $j$：
    \begin{equation*}
    \theta_{k+1}=\theta_k+\alpha^j\sqrt{\frac{2\delta}{\hat x_k^T\hat H_k\hat x_k}}\hat x_k.
    \end{equation*}
    \STATE 通过最小化价值估计与未来回报的均方误差更新 $\phi_{k+1}$。
    \ENDFOR
    \end{algorithmic}
    \end{algorithm}

接口文档
==========

.. admonition:: 你应该知道

    Spinning Up 仅提供 TRPO 的 TensorFlow 版本。以下将原网页通过 autofunction 自动引入的函数说明译为中文。

.. code-block:: python

    trpo(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=50, gamma=0.99, delta=0.01, vf_lr=1e-3,
         train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10,
         backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, logger_kwargs=dict(),
         save_freq=10, algo='trpo')

**信赖域策略优化。** 同时支持自然策略梯度（NPG）。

参数说明：

* env_fn：用于创建环境副本的函数，环境必须符合 OpenAI Gym API。
* actor_critic：接收状态占位符 x_ph 和动作占位符 a_ph 的函数，返回 TensorFlow 计算图的主要输出：pi（形状 batch, act_dim，采样动作）；logp（形状 batch，对给定动作计算对数概率）；logp_pi（形状 batch，策略采样动作的对数概率）；info（用于计算 KL 散度的分布统计量字典）；info_phs（对应旧策略统计量的占位符字典）；d_kl（当前策略和旧策略的平均 KL 散度）；v（形状 batch，价值估计，必须展平）。
* ac_kwargs（dict）：传给 actor_critic 的其他关键字参数。
* seed（int）：随机数种子。
* steps_per_epoch（int）：每轮训练中与环境交互的步数。
* epochs（int）：训练轮数，也是策略更新次数。
* gamma（float）：折扣因子，介于 0 与 1 之间。
* delta（float）：TRPO 或 NPG 更新允许的 KL 散度上限。为了稳定性通常取 0.01 或 0.05。
* vf_lr（float）：价值函数优化器学习率。
* train_v_iters（int）：每轮训练中价值函数的梯度下降更新次数。
* damping_coeff（float）：数值稳定性阻尼系数，用于将 Hessian—向量乘积调整为：

  .. math::

      Hv \rightarrow (\alpha I+H)v

  其中 :math:`\alpha` 为阻尼系数。通常不建议调整。
* cg_iters（int）：共轭梯度迭代次数。增加迭代次数可以提高 :math:`H^{-1}g` 的近似精度，但会降低运行速度。
* backtrack_iters（int）：回溯线搜索的最大回退次数。实际通常无需回退，或只回退一次。
* backtrack_coeff（float）：每次回溯缩小步长的比例，介于 0 和 1 之间，通常大于 0.5。
* lam（float）：GAE-Lambda 的 Lambda 参数，介于 0 和 1 之间且接近 1。
* max_ep_len（int）：一条轨迹／一个回合／一次 rollout 的最大长度。
* logger_kwargs（dict）：传递给 EpochLogger 的关键字参数。
* save_freq（int）：按训练轮次计的模型保存频率。
* algo：可以选择 trpo 或 npg；二者足够相似，因此共用这份代码。

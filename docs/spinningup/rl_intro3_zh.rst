========================================
第三部分：策略优化入门
========================================

.. contents:: 目录
    :depth: 2


在本节中，我们将讨论策略优化算法的数学基础，并把相关内容与示例代码联系起来。我们会介绍 **策略梯度（policy gradients）** 理论中的三个重要结果：

* 一个描述策略表现相对于策略参数的梯度的`最简单公式`_；
* 一个允许我们从表达式中`去掉无用项`_ 的规则；
* 一个允许我们向表达式中`加入有用项`_ 的规则。

最后，我们将这些结果综合起来，介绍基于优势函数的策略梯度表达式。这正是我们在`原始策略梯度（Vanilla Policy Gradient）`_ 实现中采用的版本。

.. _`最简单公式`: ../spinningup/rl_intro3.html#deriving-the-simplest-policy-gradient
.. _`去掉无用项`: ../spinningup/rl_intro3.html#don-t-let-the-past-distract-you
.. _`加入有用项`: ../spinningup/rl_intro3.html#baselines-in-policy-gradients
.. _`原始策略梯度（Vanilla Policy Gradient）`: ../algorithms/vpg.html

推导最简单的策略梯度
=====================================

在这里，我们考虑一个随机性的参数化策略 :math:`\pi_{\theta}`。我们希望最大化期望回报 :math:`J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{R(\tau)}`。为了推导方便，假设 :math:`R(\tau)` 表示`有限时域、无折扣回报`_，不过在无限时域、折扣回报情况下，推导几乎完全相同。

.. _`有限时域、无折扣回报`: ../spinningup/rl_intro.html#reward-and-return

我们希望通过梯度上升来优化策略，例如：

.. math::

    \theta_{k+1} = \theta_k + \alpha \left. \nabla_{\theta} J(\pi_{\theta}) \right|_{\theta_k}.

策略表现的梯度 :math:`\nabla_{\theta} J(\pi_{\theta})` 被称为 **策略梯度**，以这种方式优化策略的算法被称为 **策略梯度算法**。（例如原始策略梯度和 TRPO。PPO 也经常被称为策略梯度算法，尽管严格来说这种称呼略有不准确。）

要真正使用这个算法，就需要一个能够进行数值计算的策略梯度表达式。这涉及两个步骤：1）推导策略表现的解析梯度，结果会表现为一个期望；2）使用有限次智能体与环境交互所收集的数据，构造这个期望的样本估计。

本小节将推导最简单的表达式。后续小节会介绍如何改进这个最简单的版本，从而得到标准策略梯度实现实际采用的表达式。

我们先列出一些有助于推导解析梯度的事实。

**1. 一条轨迹的概率。** 如果动作来自 :math:`\pi_{\theta}`，那么轨迹 :math:`\tau = (s_0, a_0, ..., s_{T+1})` 的概率为：

.. math::

    P(\tau|\theta) = \rho_0 (s_0) \prod_{t=0}^{T} P(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t |s_t).


**2. 对数求导技巧（Log-Derivative Trick）。** 对数求导技巧基于一条简单的微积分规则：:math:`\log x` 对 :math:`x` 的导数为 :math:`1/x`。重新整理并结合链式法则，可以得到：

.. math::

    \nabla_{\theta} P(\tau | \theta) = P(\tau | \theta) \nabla_{\theta} \log P(\tau | \theta).


**3. 一条轨迹的对数概率。** 一条轨迹的对数概率为：

.. math::

    \log P(\tau|\theta) = \log \rho_0 (s_0) + \sum_{t=0}^{T} \bigg( \log P(s_{t+1}|s_t, a_t)  + \log \pi_{\theta}(a_t |s_t)\bigg).


**4. 环境函数的梯度。** 环境不依赖 :math:`\theta`，因此 :math:`\rho_0(s_0)`、:math:`P(s_{t+1}|s_t,a_t)` 和 :math:`R(\tau)` 对该参数的梯度都为零。

**5. 轨迹对数概率的梯度。** 因此，轨迹对数概率的梯度为：

.. math::

    \nabla_{\theta} \log P(\tau | \theta) &= \cancel{\nabla_{\theta} \log \rho_0 (s_0)} + \sum_{t=0}^{T} \bigg( \cancel{\nabla_{\theta} \log P(s_{t+1}|s_t, a_t)}  + \nabla_{\theta} \log \pi_{\theta}(a_t |s_t)\bigg) \\
    &= \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t).


把这些结果结合起来，可以推导得到：

.. admonition:: 基础策略梯度的推导

    .. math::
        :nowrap:

        \begin{align*}
        \nabla_{\theta} J(\pi_{\theta}) &= \nabla_{\theta} \underE{\tau \sim \pi_{\theta}}{R(\tau)} & \\
        &= \nabla_{\theta} \int_{\tau} P(\tau|\theta) R(\tau) & \text{Expand expectation} \\
        &= \int_{\tau} \nabla_{\theta} P(\tau|\theta) R(\tau) & \text{Bring gradient under integral} \\
        &= \int_{\tau} P(\tau|\theta) \nabla_{\theta} \log P(\tau|\theta) R(\tau) & \text{Log-derivative trick} \\
        &= \underE{\tau \sim \pi_{\theta}}{\nabla_{\theta} \log P(\tau|\theta) R(\tau)} & \text{Return to expectation form} \\
        \therefore \nabla_{\theta} J(\pi_{\theta}) &= \underE{\tau \sim \pi_{\theta}}{\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)} & \text{Expression for grad-log-prob}
        \end{align*}

这是一个期望值，因此我们可以使用样本均值来估计。如果收集了一组轨迹 :math:`\mathcal{D} = \{\tau_i\}_{i=1,...,N}`，每条轨迹都是智能体按照策略 :math:`\pi_{\theta}` 在环境中行动得到的，那么策略梯度就可以估计为：

.. math::

    \hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau),

其中 :math:`|\mathcal{D}|` 是 :math:`\mathcal{D}` 中的轨迹数量（即这里的 :math:`N`）。

最后这个表达式就是我们希望得到的可计算表达式的最简单版本。假设策略的表示方式允许计算 :math:`\nabla_{\theta} \log \pi_{\theta}(a|s)`，而且能够在环境中运行该策略以收集轨迹数据集，我们就可以计算策略梯度，并进行一次更新。


实现最简单的策略梯度
=========================================

我们在 `spinup/examples/pytorch/pg_math/1_simple_pg.py` 中提供了这一简单策略梯度算法的简短 PyTorch 实现。（也可以`在 GitHub 上查看 <https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py>`_。）整个文件只有 128 行，因此我们强烈推荐你深入阅读。这里不会介绍所有代码，但会重点说明并解释几个重要部分。

.. admonition:: 你应该知道

    本节以前使用 TensorFlow 示例编写。旧的 TensorFlow 内容可以在`这里 <../spinningup/extra_tf_pg_implementation.html#implementing-the-simplest-policy-gradient>`_ 找到。

**1. 构建策略网络。**

.. code-block:: python
    :linenos:
    :lineno-start: 30

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

这一代码块建立了一些模块和函数，用来使用前馈神经网络构造一个分类策略。（如果需要复习，可以参考第一部分中的`随机性策略`_。）`logits_net` 模块的输出可用于构造动作的对数概率与概率，而 `get_action` 函数根据 logits 计算出的概率采样动作。（注意：这个特定的 `get_action` 函数假设仅提供一个 `obs`，因此只输出一个整数动作。这就是它使用 `.item()` 的原因，该方法用于`取得只有一个元素的 Tensor 的内容`_。）

这里的大量工作由第 36 行的 `Categorical` 对象完成。它是 PyTorch 的一个 `Distribution` 对象，封装了一些与概率分布有关的数学函数。具体来说，它提供了从分布中采样的方法（第 40 行使用），以及计算给定样本的对数概率的方法（后面会用到）。PyTorch 的分布对象在强化学习中很有用，建议阅读`其文档`_ 来了解其工作方式。

.. admonition:: 你应该知道

    再提醒一次：当我们说一个分类分布具有“logits”时，意思是每个结果的概率由 logits 的 Softmax 函数计算。也就是说，对一个 logits 为 :math:`x_j` 的分类分布，动作 :math:`j` 的概率为：

    .. math::

        p_j = \frac{\exp(x_j)}{\sum_{i} \exp(x_i)}


.. _`随机性策略`: ../spinningup/rl_intro.html#stochastic-policies
.. _`其文档`: https://pytorch.org/docs/stable/distributions.html
.. _`取得只有一个元素的 Tensor 的内容`: https://pytorch.org/docs/stable/tensors.html#torch.Tensor.item

**2. 构建损失函数。**

.. code-block:: python
    :linenos:
    :lineno-start: 42

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()


这段代码为策略梯度算法构造了一个“损失”函数。当输入合适的数据时，这个损失的梯度等于策略梯度。合适的数据，是指在按照当前策略行动时收集的一组（状态、动作、权重）三元组，其中一个状态—动作对的权重，是它所在回合的回报。（不过，后面的小节会说明，还可以选择一些其他权重，并且同样正确。）


.. admonition:: 你应该知道

    虽然我们将它称为损失函数，但它 **不是** 监督学习中通常意义的损失函数。它与标准损失函数有两个主要区别。

    **1. 数据分布依赖于参数。** 损失函数通常定义在一个固定的数据分布上，这个分布独立于要优化的参数。而这里并非如此，因为数据必须从最新的策略中采样。

    **2. 它不衡量性能。** 损失函数通常评价我们关心的性能指标。这里我们关心的是期望回报 :math:`J(\pi_{\theta})`，但我们的“损失”函数完全不能近似它，即使从期望意义上说也是如此。这个“损失”函数对我们有用，只是因为：当在当前参数位置对其求值，且所用数据也是由当前参数生成时，它具有性能梯度的负值。

    但在第一次梯度下降更新之后，它就不再与性能有这样的联系。这意味着，对给定的一批数据进行这个“损失”函数的最小化，**绝对不能保证** 期望回报得到改善。你可以让这个损失下降到 :math:`-\infty`，同时策略性能可能一落千丈；实际上通常确实如此。深度强化学习研究者有时会把这一结果描述为策略对一批数据“过拟合”。这只是描述性的说法，不应按字面理解，因为它并不是指泛化误差。

    我们强调这一点，是因为机器学习实践者经常把损失函数视为训练中有意义的信号，认为“只要损失下降，就说明一切正常”。在策略梯度中，这种直觉是错误的，你应该只关心平均回报。损失函数本身没有意义。


.. admonition:: 你应该知道

    这里通过调用 PyTorch `Categorical` 对象的 `log_prob` 方法来生成 `logp` 张量。对于其他类型的分布对象，这种方法可能需要修改。

    例如，如果使用`正态分布`_（对应对角高斯策略），调用 `policy.log_prob(act)` 会返回一个 Tensor，其中包含每个向量动作每个分量的单独对数概率。也就是说，当你输入一个形状为 `(batch, act_dim)` 的 Tensor，会得到一个形状同样为 `(batch, act_dim)` 的 Tensor，但为了构建强化学习的损失函数，你需要的是形状为 `(batch,)` 的 Tensor。因此，你需要把动作各分量的对数概率加起来，以得到整个动作的对数概率。即：

    .. code-block:: python

        logp = get_policy(obs).log_prob(act).sum(axis=-1)


.. _`正态分布`: https://pytorch.org/docs/stable/distributions.html#normal


==========
练习
==========

.. contents:: 目录
    :depth: 2

习题集 1：实现基础
---------------------------------------

.. admonition:: 练习 1.1：高斯分布的对数似然

    **练习文件路径：**

    * PyTorch 版本：``spinup/exercises/pytorch/problem_set_1/exercise1_1.py``

    * TensorFlow 版本：``spinup/exercises/tf1/problem_set_1/exercise1_1.py``

    **参考解答路径：**

    * PyTorch 版本：``spinup/exercises/pytorch/problem_set_1_solutions/exercise1_1_soln.py``

    * TensorFlow 版本：``spinup/exercises/tf1/problem_set_1_solutions/exercise1_1_soln.py``

    **任务要求。** 编写一个函数：输入一批对角高斯分布各自的均值和标准差的对数（log std），以及先前从这些分布中生成的样本，返回这些样本的对数似然。（在 TensorFlow 版本中，你要编写一个创建相应计算图操作的函数；在 PyTorch 版本中，则直接对给定的 Tensor 执行运算。）

    回顾 `强化学习入门中的这一节`_ 所给出的公式可能会有帮助。

    请在 ``exercise1_1.py`` 中实现解答，并运行该文件以自动检查你的工作。

    **评估标准。** 使用一批随机生成的输入，将你的函数输出与已知正确的实现的输出进行比较。

.. _`强化学习入门中的这一节`: ../spinningup/rl_intro.html#stochastic-policies


.. admonition:: 练习 1.2：PPO 的策略

    **练习文件路径：**

    * PyTorch 版本：``spinup/exercises/pytorch/problem_set_1/exercise1_2.py``

    * TensorFlow 版本：``spinup/exercises/tf1/problem_set_1/exercise1_2.py``

    **参考解答路径：**

    * PyTorch 版本：``spinup/exercises/pytorch/problem_set_1_solutions/exercise1_2_soln.py``

    * TensorFlow 版本：``spinup/exercises/tf1/problem_set_1_solutions/exercise1_2_soln.py``

    **任务要求。** 为 PPO 实现一个基于多层感知机（MLP）的对角高斯策略。

    请在 ``exercise1_2.py`` 中实现解答，并运行该文件以自动检查你的工作。

    **评估标准。** 在 Gym 的 InvertedPendulum-v2 环境中运行 20 个 epoch 来评估你的解答。根据所用机器和后台运行的其他进程，实验大约需要 3–5 分钟。通过标准是：最后 5 个 epoch 的平均得分超过 500；或者在最后 5 个 epoch 中达到 1000 分（可能的最高分）。


.. admonition:: 练习 1.3：TD3 的计算图

    **练习文件路径：**

    * PyTorch 版本：``spinup/exercises/pytorch/problem_set_1/exercise1_3.py``

    * TensorFlow 版本：``spinup/exercises/tf1/problem_set_1/exercise1_3.py``

    **参考解答路径：**

    * PyTorch 版本：``spinup/algos/pytorch/td3/td3.py``

    * TensorFlow 版本：``spinup/algos/tf1/td3/td3.py``

    **任务要求。** 实现 TD3 算法的核心数学逻辑。

    所提供的起始代码包含完整的 TD3 算法，但缺少核心数学逻辑，也就是损失函数及计算这些损失所需的中间计算。请搜索 "YOUR CODE HERE"，从这些位置开始编写。

    回顾 `TD3 页面`_ 中的伪代码可能会有帮助。

    请在 ``exercise1_3.py`` 中实现解答，并运行该文件观察结果。**本练习没有自动检查。**

    **评估标准。** 运行 ``exercise1_3.py``，分别在 HalfCheetah-v2、InvertedPendulum-v2，以及另一个由你选择的 Gym MuJoCo 环境中评估代码（通过 ``--env`` 标志设置环境）。练习已配置为使用比典型 TD3 更小的神经网络（隐藏层大小为 [128,128]）、最大回合长度 150，并且只运行 10 个 epoch。目标是在较短的实际运行时间内观察到明显的学习进展。实验可能需要约 10 分钟。

    使用 ``--use_soln`` 标志可以运行 Spinning Up 自带的 TD3 实现，而非你自己实现的版本。根据经验，在 10 个 epoch 内，HalfCheetah 的得分应超过 300，而 InvertedPendulum 的得分应达到最高值 150。

.. _`TD3 页面`: ../algorithms/td3.html


习题集 2：算法的失败模式
--------------------------------------

.. admonition:: 练习 2.1：TRPO 中的价值函数拟合

    **练习文件路径。** （不适用，本练习没有代码文件。）

    **参考解答路径。** `查看练习 2.1 的解答 <../spinningup/exercise2_1_soln.html>`_。

    许多因素都可能影响策略梯度算法的性能，但很少有因素比用于优势估计的已学习价值函数的质量影响更为显著。

    在本练习中，你将比较两种 TRPO 实验的结果：一种花费大量计算精力拟合价值函数（``train_v_iters=80``），另一种几乎不花精力拟合价值函数（``train_v_iters=0``）。

    **任务要求。** 运行以下命令：

    .. parsed-literal::

        python -m spinup.run trpo --env Hopper-v2 --train_v_iters[v] 0 80 --exp_name ex2-1 --epochs 250 --steps_per_epoch 4000 --seed 0 10 20 --dt

    然后绘制结果图。（每次实验可能需要大约 10 分钟，而这条命令总共运行六次实验。）你发现了什么？

.. admonition:: 练习 2.2：DDPG 中的静默错误

    **练习文件路径：**

    * PyTorch 版本：``spinup/exercises/pytorch/problem_set_2/exercise2_2.py``

    * TensorFlow 版本：``spinup/exercises/tf1/problem_set_2/exercise2_2.py``

    **参考解答路径。** `查看练习 2.2 的解答 <../spinningup/exercise2_2_soln.html>`_。

    编写强化学习代码最困难的部分之一是处理程序错误（bug），因为失败经常悄无声息地发生。代码看起来运行正常，但智能体的性能却比没有错误的实现差，有时甚至始终学不到任何东西。

    在本练习中，你将实际观察一个 bug，并将有 bug 的结果与正确代码的结果进行比较。本练习的 PyTorch 和 TensorFlow 版本包含的 bug 在概念上相同，但具体实现不一定完全相同。

    **任务要求。** 运行 ``exercise2_2.py``，它会分别启动有 bug 和没有 bug 的 DDPG 实验。无 bug 的版本使用 Spinning Up 默认的 DDPG 实现，并采用默认方法构建 actor 和 critic 网络。有 bug 的版本运行相同的 DDPG 代码，唯一的不同是构建网络时使用了含 bug 的方法。

    总共会进行六次实验（两种情形各使用三个随机种子），每次大约需要 10 分钟。实验结束后，将结果绘制成图。有 bug 与无 bug 的性能有何区别？

    **不要参考正确的 actor-critic 代码** （即不要查看 DDPG 的 ``core.py`` 文件），尝试找出 bug 是什么，并解释它为什么会破坏算法的行为。

    **提示。** 要找出哪里出了问题，请思考 DDPG 代码如何实现 DDPG 的计算图。对于 TensorFlow 版本，可以查看以下代码片段：

    .. code-block:: python

        # Bellman backup for Q function
        backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*q_pi_targ)

        # DDPG losses
        pi_loss = -tf.reduce_mean(q_pi)
        q_loss = tf.reduce_mean((q-backup)**2)

    actor-critic 代码中的错误可能如何影响这里的计算？

    **附加挑战。** 是否存在某些超参数取值，可以掩盖这个 bug 的影响？


挑战任务
----------

.. admonition:: 从零编写代码

    按照我们在 `这篇文章 <../spinningup/spinningup.html#learn-by-doing>`_ 中的建议，尝试从零重新实现多种深度强化学习算法。

.. admonition:: 研究课题征集

    如果你已经能够熟练编写深度学习和深度强化学习代码，可以考虑尝试推进 OpenAI 长期征集的研究课题：

    * `研究课题征集 1 <https://openai.com/requests-for-research/>`_
    * `研究课题征集 2 <https://blog.openai.com/requests-for-research-2/>`_

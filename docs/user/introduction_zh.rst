============
介绍
============

.. contents:: 目录

这是什么
============

欢迎来到《Spinning Up in Deep RL》！这是 OpenAI 制作的一套教育资源，旨在让深度强化学习（deep RL）更容易入门。

对于不熟悉这一领域的读者：`强化学习`_（RL）是一种机器学习方法，通过反复试错教会智能体完成任务。深度强化学习则是强化学习与`深度学习`_ 的结合。

本模块包含多种有用资源：

* 一份简短的`强化学习导论`_，介绍术语、算法类型和基础理论；
* 一篇关于如何成长为强化学习研究人员的`文章`_；
* 一份按主题组织的重要论文`精选清单`_；
* 一个文档详尽的`代码仓库`_，提供关键算法的简短、独立实现；
* 以及若干供入门练习的`习题`_。

.. _`强化学习`: https://en.wikipedia.org/wiki/Reinforcement_learning
.. _`深度学习`: http://ufldl.stanford.edu/tutorial/

为什么创建这套资源
=================

我们最常听到的问题之一是：

    如果我想为 AI 安全作出贡献，应该如何入门？

在 OpenAI，我们认为深度学习总体上，尤其是深度强化学习，将在强大 AI 技术的发展中发挥核心作用。为了确保 AI 安全，我们必须提出与这一技术范式兼容的安全策略和算法。因此，我们鼓励提出这个问题的每个人学习这些领域。

然而，尽管已有许多资源可以帮助人们快速入门深度学习，深度强化学习的门槛仍然更高。首先，学习深度强化学习需要具备一定的数学、编程和普通深度学习基础。此外，学习者既需要从宏观层面了解这个领域：研究什么主题、为什么重要、已经取得哪些成果；也需要细致的指导，学会如何把算法理论与算法代码联系起来。

由于这一领域还很新，宏观视野并不容易建立。目前尚无一本标准的深度强化学习教材，大部分知识仍藏在论文或系列课程里，理解和消化它们往往耗费大量时间。学习实现深度强化学习算法通常也很痛苦，原因可能是：

* 发表算法的论文省略了关键设计细节，或无意间把它们讲得晦涩难懂；
* 广为传播的算法实现代码难以阅读，使人看不清代码与算法之间的对应关系。

虽然 garage_、Baselines_ 和 rllib_ 等优秀仓库让已经进入该领域的研究人员更容易推进工作，但它们将算法集成进框架时，需要作出许多并不显而易见的选择和权衡，因此不太适合直接拿来学习。结果是，深度强化学习对新研究人员、实践者和爱好者来说都有相当高的入门门槛。

所以，我们设计这套资源，就是想为那些对深度强化学习感兴趣、希望学会使用它或作出贡献，却不清楚应该学什么、如何把算法转化成代码的人，提供缺失的中间环节。我们努力让它成为尽可能有帮助的起点。

不过，实践者并不是唯一能够（或应该）从这些材料中获益的人。解决 AI 安全问题需要拥有各种专业知识和不同视角的人，其中许多相关职业与工程或计算机科学完全无关。即便如此，所有参与者仍需要充分理解相关技术，才能作出知情决策。Spinning Up 中的多个部分也旨在满足这一需要。

这些资源如何服务于我们的使命
===========================

OpenAI 的使命_ 是确保通用人工智能（AGI）的安全发展，并让更广泛的 AI 技术成果惠及大众。Spinning Up 这样的教学工具有助于推进这两个目标。

首先，每当我们帮助人们理解 AI 是什么、如何工作时，就朝着让更多人共享 AI 收益的目标迈进一步。这使人们能够批判性地思考：随着 AI 日益复杂，并在生活中越来越重要，我们预计会出现哪些问题。

同样重要的是，`我们需要人们加入`_ 确保 AGI 安全的工作。由于这一领域尚新，所需技能目前十分稀缺。我们知道许多人希望提供帮助，却不知道该怎么做：这里就是你应该学习的内容！如果能够成为这些材料所涉及领域的专家，你就有机会为 AI 安全作出贡献。

.. _`我们需要人们加入`: https://jobs.lever.co/openai

代码设计理念
======================

Spinning Up 仓库中的算法实现遵循两个设计目标：

* 在保持合理性能的前提下，尽可能简单；
* 不同算法之间保持高度一致，以展示它们在基本原理上的相似之处。

这些实现几乎完全独立，彼此基本不共享代码（日志记录、保存、加载及 `MPI`_ 工具除外）。因此，感兴趣的读者可以单独研究每个算法，而不必沿着没完没了的依赖链追踪其实现。代码尽可能贴近伪代码，以缩小理论与代码之间的距离。

更重要的是，所有算法的代码结构都很相似。因此，只要彻底理解其中一种算法，转向学习下一种就不会太困难。

我们尽可能减少各算法实现中使用的技巧，并尽量缩小相似算法之间的差异。例如，我们删去了原始 Soft Actor-Critic 代码中的`正则化项`_，也从所有算法中移除了`观测归一化`_。另一个减少算法间差异的例子是：我们实现的 DDPG、TD3 和 SAC 都遵循在固定环境交互间隔之后进行梯度下降更新的约定。（相比之下，这些算法的其他公开实现通常采用彼此略有不同的方法，因此更难比较。）

所有算法都属于“性能合理”的实现：它们大致能够达到预期表现，但不一定在所有任务上匹配文献报道的最佳结果。因此，使用这些实现进行科学基准比较时务必谨慎。各实现具体的性能水平详见我们的`基准测试`_ 页面。

.. _`MPI`: https://en.wikipedia.org/wiki/Message_Passing_Interface

长期维护与支持历史
=====================================

Spinning Up 目前处于维护模式。如果发现会导致功能失效的严重错误，我们会修复，以确保它仍能帮助人们学习深度强化学习。

迄今为止的支持历史：

* **2018 年 11 月 8 日：** 首次发布！

* **2018 年 11 月：** 发布后提供了为期三周的高强度支持。

* **2019 年 4 月：** 发布约六个月后，我们根据社区反馈对 Spinning Up 进行了内部评估。评估发现，人们对几项重要功能很感兴趣：

  * **使用其他神经网络库实现算法。** 一些人希望看到使用 TensorFlow v1 以外工具实现的强化学习算法。部分社区成员甚至自行开发了 Spinning Up 算法的 PyTorch 版本，例如 Kashif Rasul 的 `Fired Up`_、Kai Arulkumaran 的 `Spinning Up Basic`_ 和 Misha Laskin 的 `Torching Up`_。因此，为深度强化学习打造这种“罗塞塔石碑”式的跨框架对照资源，成为后续工作的高优先级事项。

  * **开源强化学习环境。** 许多人希望 Spinning Up 在基准测试、示例和练习中使用更多开源强化学习环境，例如 `PyBullet`_。

  * **更多算法。** 一些人希望 Spinning Up 纳入其他算法，尤其是深度 Q 网络（DQN）。

* **2020 年 1 月：** 发布了 Spinning Up 的 PyTorch 更新！

* **未来：** 目前没有计划对 Spinning Up 进行重大更新。如果将来有理由再次更新，那么按照六个月评估的结论，下一优先事项将是更多使用开源强化学习环境，以及添加更多算法。

此外，正如博客文章所讨论的，Spinning Up 已被纳入我们的 Scholars_ 和 Fellows_ 项目课程。

.. _`强化学习导论`: ../spinningup/rl_intro.html
.. _`文章`: ../spinningup/spinningup.html
.. _`精选清单`: ../spinningup/keypapers.html
.. _`代码仓库`: https://github.com/openai/spinningup
.. _`习题`: ../spinningup/exercises.html
.. _garage: https://github.com/rlworkgroup/garage
.. _Baselines: https://github.com/openai/baselines
.. _rllib: https://github.com/ray-project/ray/tree/master/python/ray/rllib
.. _使命: https://blog.openai.com/openai-charter/
.. _`正则化项`: https://github.com/haarnoja/sac/blob/108a4229be6f040360fcca983113df9c4ac23a6a/sac/distributions/normal.py#L69
.. _`观测归一化`: https://github.com/openai/baselines/blob/28aca637d0f13f4415cc5ebb778144154cff3110/baselines/run.py#L131
.. _`原始 TD3 代码`: https://github.com/sfujim/TD3/blob/25dfc0a6562c54ae5575fad5b8f08bc9d5c4e26c/main.py#L89
.. _`基准测试`: ../spinningup/bench.html
.. _`Fired Up`: https://github.com/kashif/firedup
.. _`Spinning Up Basic`: https://github.com/Kaixhin/spinning-up-basic
.. _`Torching Up`: https://github.com/MishaLaskin/torchingup
.. _`PyBullet`: https://pybullet.org/wordpress/
.. _`MuJoCo`: http://mujoco.org/
.. _Scholars: https://openai.com/blog/openai-scholars-spring-2020/
.. _Fellows: https://openai.com/blog/openai-fellows-fall-2018/

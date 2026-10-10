===================================
如何成为一名深度强化学习研究者
===================================
Joshua Achiam 撰，2018 年 10 月 13 日

.. contents:: 目录
    :depth: 2

如果你立志成为一名深度强化学习（deep RL）研究者，到现在大概已经听过关于深度强化学习的各种说法。你知道 `它很难，而且并不总是有效`_。即使严格按照现成的方法操作，`可复现性`_ `仍然是一项挑战`_。如果从零开始，`学习曲线也极其陡峭`_。与此同时，确实有许多 `优秀的`_ `学习资源`_ `散布`_ `各处`_，但由于这一领域的材料仍然相对新颖，尚不存在一条明确、经过充分探索的精通之路。本专栏旨在帮助你跨越最初的门槛，并明确应该如何起步，成为一名深度强化学习研究者。具体来说，文章会勾勒一份有助于增长基础知识的学习计划，同时穿插那些能让研究做得更好的其他实践经验。

合适的知识背景
====================

**打好扎实的数学基础。** 对概率论和统计学，你应该能够熟练运用随机变量、贝叶斯定理、概率的链式法则、期望、标准差和重要性采样。对多元微积分，你需要理解梯度，以及泰勒级数展开（后者并非必需，但掌握它会有帮助）。

**建立对深度学习的一般认识。** 你不必掌握每一个特殊技巧或每一种网络架构，但基本知识会很有帮助。应该了解常见架构，包括 `MLP`_、`vanilla RNN`_、`LSTM`_ （ `也可以读这篇博客`_ ）、`GRU`_、`卷积`_ `层`_、`ResNet`_ 和 `注意力`_ `机制`_；常见正则化方法，包括 `权重衰减`_ 和 `dropout`_；归一化方法，包括 `批归一化`_、`层归一化`_ 和 `权重归一化`_；以及优化器，包括 `SGD、带动量的 SGD`_、`Adam`_ 和 `其他方法`_。还应该知道什么是 `重参数化技巧`_。

**熟悉至少一个深度学习软件库。** `TensorFlow`_ 或 `PyTorch`_ 都是不错的起点。你不需要知道如何完成所有任务，但应该对编写一个简单的监督学习程序有相当充分的信心。

**熟悉强化学习的主要概念和术语。** 你应该知道状态、动作、轨迹、策略、奖励、价值函数和动作价值函数分别是什么。如果不熟悉，Spinning Up 提供了介绍这些内容的 `入门章节`_；还值得看看 OpenAI 黑客松的 `RL-Intro`_，或者 Lilian Weng 撰写的出色而详尽的 `强化学习综述`_。如果你喜欢数学理论，还可以选学 `单调改进理论`_ （它是高级策略梯度算法的基础）的数学知识，或 `经典强化学习算法`_ （虽然它们已被深度强化学习算法超越，却包含有价值的见解，有时还能推动新研究）。

在实践中学习
==============

**亲自编写算法实现。** 你应该尽可能从零实现更多核心深度强化学习算法，并以为每个算法写出最简短且正确的实现为目标。这是理解算法运行原理、培养对各算法具体性能特征的直觉的最佳方式，远胜其他方法。

**简单至关重要。** 应该安排好实践顺序，先实现最简单的算法，再逐渐引入复杂性。如果一开始就尝试构建包含过多相互作用部分的系统，它很可能出错，并让你花上几个星期调试。这是深度强化学习初学者常见的失败模式。如果陷入这种困境，不必灰心，但应尝试调整方向，先完成更简单的算法，再回来处理复杂的那个。

**应该实现哪些算法？** 大致按以下顺序开始比较合适：基础策略梯度（也称为 `REINFORCE`_）、`DQN`_、`A2C`_ （ `A3C`_ 的同步版本）、`PPO`_ （采用裁剪目标函数的变体）以及 `DDPG`_。这些算法最简单的实现都只需要几百行代码（大约 250 至 300 行），其中一些甚至更少（例如，一个 `不加额外技巧的 VPG 版本`_ 只需约 80 行）。尝试编写这些算法的并行版本之前，应先写出单线程代码。（不过，至少应该尝试把其中一个算法并行化。）

**重点放在理解上。** 编写能够正常学习的强化学习代码，需要对算法有清晰、细致的理解。这是因为 **有问题的强化学习代码几乎总会悄无声息地失败**：代码看起来运行正常，但智能体始终学不会如何完成任务。问题通常出在某个量使用了错误的公式计算，或者基于错误的分布计算，或者把数据送到了错误的位置。有时，找出这类错误的唯一办法，就是以批判性的眼光阅读代码，明确知道代码本应执行什么，再找出它偏离正确行为的地方。要形成这样的知识，就必须研读学术文献，并在可能的情况下参考其他已有实现，因此应该把相当一部分时间用于这类阅读。

**读论文时要重点查看什么？** 根据论文实现算法时，要把论文仔细研究透彻，尤其是消融分析以及补充材料（如果有的话）。消融实验能帮助你形成直觉，判断哪些参数或子程序对算法能否正常工作影响最大，从而帮助诊断错误。补充材料通常会提供网络架构、优化超参数等具体细节；你应尽量让自己的实现与这些细节保持一致，以提高实现成功的机会。

**但不要过度拘泥于论文细节。** 有时论文会规定使用比真正必要的技巧更多的技巧，因此应该对此保持警惕，并在可能时尝试简化。例如，最初的 DDPG 论文建议使用复杂的神经网络架构、初始化方案和批归一化。这些并非严格必需，而且一些报告表现最好的 DDPG 实验使用了更简单的网络。再如，最初的 A3C 论文使用来自多个 actor-learner 的异步更新，但后来发现同步更新的效果也差不多好。

**也不要过度照搬已有实现。** 可以研究 `已有`_ `实现`_ 来获得启发，但不要过度依赖其中的工程细节。强化学习库经常为了算法之间的代码复用而选择某些抽象方式，这对于复用确实有好处；但如果你只实现一种算法，或只支持单一使用场景，这些抽象可能毫无必要。

**在简单环境中快速迭代。** 调试自己的算法实现时，应先尝试那些能够快速学会任务的简单环境，例如 `OpenAI Gym`_ 提供的 CartPole-v0、InvertedPendulum-v0、FrozenLake-v0 和 HalfCheetah-v2。对于 HalfCheetah-v2，应使用较短的时间范围，即只运行 100 或 250 步，而不是完整的 1000 步。如果连最简单的玩具任务是否能学会都还没验证，就不要直接尝试 Atari 或复杂的 Humanoid 环境。在调试阶段，理想的单次实验周转时间应少于 5 分钟（在你的本地机器上），也可以稍长一点，但不能长太多。这些小规模实验不需要特殊硬件，在 CPU 上就能相对轻松地运行。

**如果算法不起作用，先假定存在错误。** 在诉诸超参数调整之前，应花大量精力查找程序错误，因为通常就是程序有问题。糟糕的超参数的确可能严重损害强化学习性能，但如果你使用的超参数与论文或标准实现中的相近，问题很可能不出在它们身上。另外请记住：即使存在会导致失败的程序错误，算法有时仍可能在某个环境中正常工作。因此，一旦结果看起来有希望，也务必要在不止一个环境中测试。

**测量一切。** 应添加大量监测手段，了解算法内部到底发生了什么。每次迭代都能查看的学习过程统计信息越多，调试就越容易。毕竟，如果无法看到哪里出问题，就无法判断它正在出问题。我个人喜欢查看累计奖励、回合长度和价值函数估计的均值、标准差、最小值及最大值，也会查看目标函数的损失，以及各种探索参数的细节（如优化随机策略时的平均熵，或 DQN 采用 ε-贪心策略时的当前 ε 值）。此外，还应该不时观看智能体表现的视频；这会带来单靠其他手段无法得到的见解。

**算法有效后再扩大实验规模。** 当你已经实现某种强化学习算法，且它看起来能在最简单的环境中正确运行，就应该开始在更难的环境中测试。这一阶段的实验会花更长时间，视具体情况可能需要数小时到几天。此时，强大的 GPU 或 32 核机器等专门硬件可能很有用，也应该考虑使用 AWS 或 GCE 等云计算资源。

**继续保持这些习惯！** 这些习惯的价值不只限于学习深度强化学习的阶段，它们还会加快你今后的研究进展！

开展研究项目
=============================

当你对深度强化学习的基础知识已经比较熟悉，就应该开始探索领域边界并开展研究。为此，你需要一个项目构想。

**首先阅读文献，了解领域内有哪些研究主题。** 你可能会对许多方向感兴趣，例如样本效率、探索、迁移学习、层次化、记忆、基于模型的强化学习、元学习和多智能体等。如果正在寻找灵感，或者只想大致了解已有工作，可以查阅 Spinning Up 的 `关键论文列表 <../spinningup/keypapers.html>`_。从这些主题中挑一篇你喜欢、能够激发兴趣的论文，认真通读。利用其相关工作章节和参考文献，找到紧密相关的论文，深入梳理研究文献。你将逐渐发现尚未解决的问题，以及自己可能在哪些地方产生影响。

**产生研究想法的不同切入角度：** 思考研究项目的方法有很多种，而你选择的切入角度，会影响项目后续如何发展以及面临何种风险。下面是几个例子：

**角度一：改进现有方法。** 这是一种渐进式研究路线：通过调整已有算法，在成熟的问题设定中争取更好的性能。在这里，重新实现前人的工作非常有帮助，因为你会亲自发现已有算法在哪些方面脆弱，以及可能如何改进。对新手来说，这是最容易进入的研究路线，但对任何经验水平的研究者也都有价值。尽管有些研究者认为渐进式改进不那么令人兴奋，但机器学习中一些最令人印象深刻的成就就来自这种工作。

由于这类项目依附于已有方法，其范围天然较窄，通常可以比较快地完成（几个月），这可能正合你的需要，尤其是刚开始做研究时。不过，这也带来了风险：你对算法设想的改动有可能无法改善性能。那样一来，如果想不出其他改动，项目就只能结束，而且你没有明确的下一步方向。

**角度二：聚焦尚未解决的基准任务。** 与其思考如何改进已有方法，不如尝试解决一个至今没人解决的任务。例如，在 `Sonic 任务领域`_ 或 `Gym Retro`_ 中，实现从训练关卡到测试关卡的完美泛化。为攻克一个未解决的任务，你可能会尝试各种各样的方法，包括现有方法和专门为该项目发明的新方法。新手也能尝试这一类问题，但学习曲线会更陡峭。

这种项目的范围较广，可能持续相当长时间（数月到一年以上）。主要风险是，若没有重大突破，基准任务可能根本无法解决，这意味着你很容易投入大量时间却毫无进展。但即使这类项目最终失败，它通常也会让研究者获得许多新的见解，为下一个项目提供丰富的土壤。

**角度三：创造一种新的问题设定。** 不要局限于已有方法或当前重大挑战，而是思考一个尚未被研究的、概念上完全不同的问题，再设法在该问题上取得进展。沿着这种思路开展的项目，通常还没有标准基准，因此你必须亲自设计一个。这可能极具挑战性，但值得迎接这个挑战，因为优秀的基准任务能够推动整个领域前进。

这类问题往往是在某个时机自然出现的，很难刻意寻找。

**避免重复造轮子。** 当你想到一个想要测试的好主意时，那当然值得高兴！但在项目仍处于早期阶段时，应该尽可能全面地检查它是否已经被别人做过。项目做到一半才发现早已有论文提出相同想法，会让人非常沮丧。如果恰好遇到同期独立开展的研究，就更令人懊恼，而这种情况确实偶尔会发生。不过，不要因此退缩，更不要因此急着用尚未完成的研究抢先宣示成果，夸大阶段性工作的价值。应该做好研究，以完整、彻底的调查完成项目，因为这才是真正重要的事，从长期来看尤其如此。

在强化学习中开展严谨研究
=============================

现在，你已经想到了一个研究构想，而且比较确信别人尚未做过。你运用已掌握的技能把它实现出来，并开始在标准任务上测试。它看起来有效！但这意味着什么？效果必须好到什么程度，才能算是一项重要成果？这是深度强化学习研究中最困难的问题之一。为了验证你的方法确实作出了有意义的贡献，你必须严谨地证明它在性能上超过尽可能强的基线算法，也就是当前在测试任务上达到 SOTA（state of the art，最先进水平）的算法。如果你发明了新的测试任务，因此没有既有的 SOTA，仍须尝试文献中那些最可靠、且合理预期可能在新任务上表现良好的算法，并超越它们。

**建立公平的比较。** 如果基线算法是你从零实现的，而不是直接拿另一篇论文报告的数值来比较，那么用于调优基线的时间应当与调优自己算法的时间一样多，这样才能确保比较公平。即使你的算法与基线之间存在显著差异，也应该尽最大努力保持“其他条件相同”。例如，若研究的是网络架构的不同变体，就应该让自己模型与基线模型的参数数量大致相等。无论如何都不能故意让基线处于不利地位！事实上，强化学习中的基线算法往往非常强，要稳定且显著地超越它们可能十分困难，或需要在算法设计上提出有价值的见解。

**排除随机性这一混杂因素。** 警惕随机种子使结果看起来比实际更强或更弱，因此所有实验都应运行多个随机种子（至少 3 个，如果想做得更彻底，就运行 10 个或更多）。这一点非常重要，值得反复强调：在许多常见应用中，深度强化学习对随机种子似乎相当敏感。结果的方差可能大到这种程度：使用两组不同随机种子得到的学习曲线之间差异极其明显，看起来甚至完全不像来自同一个分布（参见 `这里的图 10`_）。

**开展高诚信度的实验。** 不要只挑选表现最好或最有意思的几次运行结果写进论文。相反，你应该为准备比较的所有方法重新开展最终实验（如果是比较自己实现的基线算法），并事先承诺如实报告这些最终实验得到的结果。这相当于实行一种较弱形式的 `预注册`_：在调优阶段形成假设，在最终实验阶段得出结论。

**分别检验每项主张。** 开展研究的另一个关键环节是消融分析。你提出的任何方法，很可能都包含几项关键设计决定，例如架构选择或正则化技术，而每一项都可能单独影响性能。你在论文中声称的是，这些设计决定共同带来了帮助，但实际上这相当于把多项主张打包在一起：每一个设计元素都对应一项主张。通过系统评估将某个元素替换为其他设计、或彻底移除时会发生什么，就能正确判断方法收益应归功于哪些部分。这使你能够以一定的信心分别提出各项主张，并增强整体研究的说服力。

结束语
================

深度强化学习是一个令人兴奋、发展迅速的领域。我们需要尽可能多的人参与解决开放问题，并推动研究进步。希望读完本文，你已经为成为其中的一员做好了更多准备！当你准备好时，`告诉我们`_。

.. _`告诉我们`: https://jobs.lever.co/openai

附录：其他资源
===================

还可以阅读以下文章，进一步了解如何在这一领域成长为研究者或工程师：

`Advice for Short-term Machine Learning Research Projects <https://rockt.github.io/2018/08/29/msc-advice>`_，作者 Tim Rocktäschel、Jakob Foerster 和 Greg Farquhar。

`ML Engineering for AI Safety & Robustness: a Google Brain Engineer’s Guide to Entering the Field <https://80000hours.org/articles/ml-engineering-career-transition-guide/>`_，作者 Catherine Olsson 和 80,000 Hours。

参考文献
========


.. _`它很难，而且并不总是有效`: https://www.alexirpan.com/2018/02/14/rl-hard.html
.. [1] `Deep Reinforcement Learning Doesn't Work Yet <https://www.alexirpan.com/2018/02/14/rl-hard.html>`_, Alex Irpan, 2018

.. _`可复现性`: https://arxiv.org/abs/1708.04133
.. _`这里的图 10`: https://arxiv.org/pdf/1708.04133.pdf
.. [2] `Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control <https://arxiv.org/abs/1708.04133>`_, Islam et al, 2017

.. _`仍然是一项挑战`: https://arxiv.org/abs/1709.06560
.. [3] `Deep Reinforcement Learning that Matters <https://arxiv.org/abs/1709.06560>`_, Henderson et al, 2017

.. _`学习曲线也极其陡峭`: http://amid.fish/reproducing-deep-rl
.. [4] `Lessons Learned Reproducing a Deep Reinforcement Learning Paper <http://amid.fish/reproducing-deep-rl>`_, Matthew Rahtz, 2018

.. _`优秀的`: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html
.. [5] `UCL Course on RL <http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html>`_

.. _`学习资源`: http://rll.berkeley.edu/deeprlcourse/
.. [6] `Berkeley Deep RL Course <http://rll.berkeley.edu/deeprlcourse/>`_

.. _`散布`: https://sites.google.com/view/deep-rl-bootcamp/lectures
.. [7] `Deep RL Bootcamp <https://sites.google.com/view/deep-rl-bootcamp/lectures>`_

.. _`各处`: http://joschu.net/docs/nuts-and-bolts.pdf
.. [8] `Nuts and Bolts of Deep RL <http://joschu.net/docs/nuts-and-bolts.pdf>`_, John Schulman

.. _`MLP`: http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/
.. [9] `Stanford Deep Learning Tutorial: Multi-Layer Neural Network <http://ufldl.stanford.edu/tutorial/supervised/MultiLayerNeuralNetworks/>`_

.. _`Vanilla RNN`: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
.. [10] `The Unreasonable Effectiveness of Recurrent Neural Networks <http://karpathy.github.io/2015/05/21/rnn-effectiveness/>`_, Andrej Karpathy, 2015

.. _`LSTM`: https://arxiv.org/abs/1503.04069
.. [11] `LSTM: A Search Space Odyssey <https://arxiv.org/abs/1503.04069>`_, Greff et al, 2015

.. _`也可以读这篇博客`: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
.. [12] `Understanding LSTM Networks <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>`_, Chris Olah, 2015

.. _`GRU`: https://arxiv.org/abs/1412.3555v1
.. [13] `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling <https://arxiv.org/abs/1412.3555v1>`_, Chung et al, 2014 (GRU paper)

.. _`卷积`: http://colah.github.io/posts/2014-07-Conv-Nets-Modular/
.. [14] `Conv Nets: A Modular Perspective <http://colah.github.io/posts/2014-07-Conv-Nets-Modular/>`_, Chris Olah, 2014

.. _`层`: https://cs231n.github.io/convolutional-networks/
.. [15] `Stanford CS231n, Convolutional Neural Networks for Visual Recognition <https://cs231n.github.io/convolutional-networks/>`_

.. _`ResNet`: https://arxiv.org/abs/1512.03385
.. [16] `Deep Residual Learning for Image Recognition <https://arxiv.org/abs/1512.03385>`_, He et al, 2015 (ResNets)

.. _`注意力`: https://arxiv.org/abs/1409.0473
.. [17] `Neural Machine Translation by Jointly Learning to Align and Translate <https://arxiv.org/abs/1409.0473>`_, Bahdanau et al, 2014 (Attention mechanisms)

.. _`机制`: https://arxiv.org/abs/1706.03762
.. [18] `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_, Vaswani et al, 2017

.. _`权重衰减`: https://papers.nips.cc/paper/563-a-simple-weight-decay-can-improve-generalization.pdf
.. [19] `A Simple Weight Decay Can Improve Generalization <https://papers.nips.cc/paper/563-a-simple-weight-decay-can-improve-generalization.pdf>`_, Krogh and Hertz, 1992


.. _`dropout`: http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf
.. [20] `Dropout:  A Simple Way to Prevent Neural Networks from Overfitting <http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf>`_, Srivastava et al, 2014

.. _`批归一化`: https://arxiv.org/abs/1502.03167
.. [21] `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_, Ioffe and Szegedy, 2015

.. _`层归一化`: https://arxiv.org/abs/1607.06450
.. [22] `Layer Normalization <https://arxiv.org/abs/1607.06450>`_, Ba et al, 2016

.. _`权重归一化`: https://arxiv.org/abs/1602.07868
.. [23] `Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks <https://arxiv.org/abs/1602.07868>`_, Salimans and Kingma, 2016

.. _`SGD、带动量的 SGD`: http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/
.. [24] `Stanford Deep Learning Tutorial: Stochastic Gradient Descent <http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/>`_

.. _`Adam`: https://arxiv.org/abs/1412.6980
.. [25] `Adam: A Method for Stochastic Optimization <https://arxiv.org/abs/1412.6980>`_, Kingma and Ba, 2014

.. _`其他方法`: https://arxiv.org/abs/1609.04747
.. [26] `An overview of gradient descent optimization algorithms <https://arxiv.org/abs/1609.04747>`_, Sebastian Ruder, 2016

.. _`重参数化技巧`: https://arxiv.org/abs/1312.6114
.. [27] `Auto-Encoding Variational Bayes <https://arxiv.org/abs/1312.6114>`_, Kingma and Welling, 2013 (Reparameterization trick)

.. _`TensorFlow`: https://www.tensorflow.org/
.. [28] `Tensorflow`_

.. _`PyTorch`: http://pytorch.org/
.. [29] `PyTorch`_

.. _`入门章节`: ../spinningup/rl_intro.html
.. [30] `Spinning Up in Deep RL: Introduction to RL, Part 1 <../spinningup/rl_intro.html>`_

.. _`RL-Intro`: https://github.com/jachiam/rl-intro/blob/master/Presentation/rl_intro.pdf
.. [31] `RL-Intro`_ Slides from OpenAI Hackathon, Josh Achiam, 2018

.. _`强化学习综述`: https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html
.. [32] `A (Long) Peek into Reinforcement Learning <https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html>`_, Lilian Weng, 2018

.. _`单调改进理论`: http://joschu.net/docs/thesis.pdf
.. [33] `Optimizing Expectations <http://joschu.net/docs/thesis.pdf>`_, John Schulman, 2016 (Monotonic improvement theory)

.. _`经典强化学习算法`: https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf
.. [34] `Algorithms for Reinforcement Learning <https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf>`_, Csaba Szepesvari, 2009 (Classic RL Algorithms)

.. _`REINFORCE`: https://arxiv.org/abs/1604.06778
.. [35] `Benchmarking Deep Reinforcement Learning for Continuous Control <https://arxiv.org/abs/1604.06778>`_, Duan et al, 2016

.. _`DQN`: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
.. [36] `Playing Atari with Deep Reinforcement Learning <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_, Mnih et al, 2013 (DQN)

.. _`A2C`: https://blog.openai.com/baselines-acktr-a2c/
.. [37] `OpenAI Baselines: ACKTR & A2C <https://blog.openai.com/baselines-acktr-a2c/>`_

.. _`A3C`: https://arxiv.org/abs/1602.01783
.. [38] `Asynchronous Methods for Deep Reinforcement Learning <https://arxiv.org/abs/1602.01783>`_, Mnih et al, 2016 (A3C)

.. _`PPO`: https://arxiv.org/abs/1707.06347
.. [39] `Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_, Schulman et al, 2017 (PPO)

.. _`DDPG`: https://arxiv.org/abs/1509.02971
.. [40] `Continuous Control with Deep Reinforcement Learning <https://arxiv.org/abs/1509.02971>`_, Lillicrap et al, 2015 (DDPG)

.. _`不加额外技巧的 VPG 版本`: https://github.com/jachiam/rl-intro/blob/master/pg_cartpole.py
.. [41] `RL-Intro Policy Gradient Sample Code <https://github.com/jachiam/rl-intro/blob/master/pg_cartpole.py>`_, Josh Achiam, 2018

.. _`已有`: https://github.com/openai/baselines
.. [42] `OpenAI Baselines <https://github.com/openai/baselines>`_

.. _`实现`: https://github.com/rll/rllab
.. [43] `rllab <https://github.com/rll/rllab>`_

.. _`OpenAI Gym`: https://gym.openai.com/
.. [44] `OpenAI Gym <https://gym.openai.com/>`_

.. _`Sonic 任务领域`: https://contest.openai.com/2018-1/
.. [45] `OpenAI Retro Contest <https://contest.openai.com/2018-1/>`_

.. _`Gym Retro`: https://blog.openai.com/gym-retro/
.. [46] `OpenAI Gym Retro <https://blog.openai.com/gym-retro/>`_

.. _`预注册`: https://cos.io/prereg/
.. [47] `Center for Open Science <https://cos.io/prereg/>`_, explaining what preregistration means in the context of scientific experiments.
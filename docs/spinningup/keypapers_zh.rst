======================
深度强化学习关键论文
======================

以下列出一些值得阅读的深度强化学习论文。这份清单 *远非* 详尽无遗，但可以为希望在这一领域开展研究的人提供一个有用的起点。

.. contents:: 目录
    :depth: 2


1. 无模型强化学习
===================

a. 深度 Q 学习
------------------


.. [#] `Playing Atari with Deep Reinforcement Learning <https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf>`_, Mnih et al, 2013. **算法： DQN.**

.. [#] `Deep Recurrent Q-Learning for Partially Observable MDPs <https://arxiv.org/abs/1507.06527>`_, Hausknecht and Stone, 2015. **算法： Deep Recurrent Q-Learning.**

.. [#] `Dueling Network Architectures for Deep Reinforcement Learning <https://arxiv.org/abs/1511.06581>`_, Wang et al, 2015. **算法： Dueling DQN.**

.. [#] `Deep Reinforcement Learning with Double Q-learning <https://arxiv.org/abs/1509.06461>`_, Hasselt et al 2015. **算法： Double DQN.**

.. [#] `Prioritized Experience Replay <https://arxiv.org/abs/1511.05952>`_, Schaul et al, 2015. **算法： Prioritized Experience Replay (PER).**

.. [#] `Rainbow: Combining Improvements in Deep Reinforcement Learning <https://arxiv.org/abs/1710.02298>`_, Hessel et al, 2017. **算法： Rainbow DQN.**


b. 策略梯度
-------------------


.. [#] `Asynchronous Methods for Deep Reinforcement Learning <https://arxiv.org/abs/1602.01783>`_, Mnih et al, 2016. **算法： A3C.**

.. [#] `Trust Region Policy Optimization <https://arxiv.org/abs/1502.05477>`_, Schulman et al, 2015. **算法： TRPO.**

.. [#] `High-Dimensional Continuous Control Using Generalized Advantage Estimation <https://arxiv.org/abs/1506.02438>`_, Schulman et al, 2015. **算法： GAE.**

.. [#] `Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_, Schulman et al, 2017. **算法： PPO-Clip, PPO-Penalty.**

.. [#] `Emergence of Locomotion Behaviours in Rich Environments <https://arxiv.org/abs/1707.02286>`_, Heess et al, 2017. **算法： PPO-Penalty.**

.. [#] `Scalable trust-region method for deep reinforcement learning using Kronecker-factored approximation <https://arxiv.org/abs/1708.05144>`_, Wu et al, 2017. **算法： ACKTR.**

.. [#] `Sample Efficient Actor-Critic with Experience Replay <https://arxiv.org/abs/1611.01224>`_, Wang et al, 2016. **算法： ACER.**

.. [#] `Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor <https://arxiv.org/abs/1801.01290>`_, Haarnoja et al, 2018. **算法： SAC.**

c. 确定性策略梯度
---------------------------------


.. [#] `Deterministic Policy Gradient Algorithms <http://proceedings.mlr.press/v32/silver14.pdf>`_, Silver et al, 2014. **算法： DPG.**

.. [#] `Continuous Control With Deep Reinforcement Learning <https://arxiv.org/abs/1509.02971>`_, Lillicrap et al, 2015. **算法： DDPG.**

.. [#] `Addressing Function Approximation Error in Actor-Critic Methods <https://arxiv.org/abs/1802.09477>`_, Fujimoto et al, 2018. **算法： TD3.**


d. 分布强化学习
--------------------

.. [#] `A Distributional Perspective on Reinforcement Learning <https://arxiv.org/abs/1707.06887>`_, Bellemare et al, 2017. **算法： C51.** 

.. [#] `Distributional Reinforcement Learning with Quantile Regression <https://arxiv.org/abs/1710.10044>`_, Dabney et al, 2017. **算法： QR-DQN.**

.. [#] `Implicit Quantile Networks for Distributional Reinforcement Learning <https://arxiv.org/abs/1806.06923>`_, Dabney et al, 2018. **算法： IQN.**

.. [#] `Dopamine: A Research Framework for Deep Reinforcement Learning <https://openreview.net/forum?id=ByG_3s09KX>`_, Anonymous, 2018. **贡献： 介绍 Dopamine 代码仓库，其中包含 DQN、C51、IQN 和 Rainbow 的实现。 `代码链接 <https://github.com/google/dopamine>`_

e. 使用依赖动作基线的策略梯度
---------------------------------------------------

.. [#] `Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic <https://arxiv.org/abs/1611.02247>`_, Gu et al, 2016. **算法： Q-Prop.**

.. [#] `Action-depedent Control Variates for Policy Optimization via Stein's Identity <https://arxiv.org/abs/1710.11198>`_, Liu et al, 2017. **算法： Stein Control Variates.**

.. [#] `The Mirage of Action-Dependent Baselines in Reinforcement Learning <https://arxiv.org/abs/1802.10031>`_, Tucker et al, 2018. **贡献： 对早期论文（包括 Q-Prop 和 Stein 控制变量方法）的主张提出批评并重新评估，发现了其中重要的方法学错误。这一点尤其值得注意。


f. 路径一致性学习
----------------------------

.. [#] `Bridging the Gap Between Value and Policy Based Reinforcement Learning <https://arxiv.org/abs/1702.08892>`_, Nachum et al, 2017. **算法： PCL.**

.. [#] `Trust-PCL: An Off-Policy Trust Region Method for Continuous Control <https://arxiv.org/abs/1707.01891>`_, Nachum et al, 2017. **算法： Trust-PCL.**

g. 结合策略学习与 Q 学习的其他方向
----------------------------------------------------------------

.. [#] `Combining Policy Gradient and Q-learning <https://arxiv.org/abs/1611.01626>`_, O'Donoghue et al, 2016. **算法： PGQL.**

.. [#] `The Reactor: A Fast and Sample-Efficient Actor-Critic Agent for Reinforcement Learning <https://arxiv.org/abs/1704.04651>`_, Gruslys et al, 2017. **算法： Reactor.**

.. [#] `Interpolated Policy Gradient: Merging On-Policy and Off-Policy Gradient Estimation for Deep Reinforcement Learning <http://papers.nips.cc/paper/6974-interpolated-policy-gradient-merging-on-policy-and-off-policy-gradient-estimation-for-deep-reinforcement-learning>`_, Gu et al, 2017. **算法： IPG.**

.. [#] `Equivalence Between Policy Gradients and Soft Q-Learning <https://arxiv.org/abs/1704.06440>`_, Schulman et al, 2017. **贡献： 揭示这两类强化学习算法之间的理论联系。


h. 进化算法
--------------------------

.. [#] `Evolution Strategies as a Scalable Alternative to Reinforcement Learning <https://arxiv.org/abs/1703.03864>`_, Salimans et al, 2017. **算法： ES.**



2. 探索
==============

a. 内在动机
-----------------------

.. [#] `VIME: Variational Information Maximizing Exploration <https://arxiv.org/abs/1605.09674>`_, Houthooft et al, 2016. **算法： VIME.**

.. [#] `Unifying Count-Based Exploration and Intrinsic Motivation <https://arxiv.org/abs/1606.01868>`_, Bellemare et al, 2016. **算法： CTS-based Pseudocounts.**

.. [#] `Count-Based Exploration with Neural Density Models <https://arxiv.org/abs/1703.01310>`_, Ostrovski et al, 2017. **算法： PixelCNN-based Pseudocounts.**

.. [#] `#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning <https://arxiv.org/abs/1611.04717>`_, Tang et al, 2016. **算法： Hash-based Counts.**

.. [#] `EX2: Exploration with Exemplar Models for Deep Reinforcement Learning <https://arxiv.org/abs/1703.01260>`_, Fu et al, 2017. **算法： EX2.**

.. [#] `Curiosity-driven Exploration by Self-supervised Prediction <https://arxiv.org/abs/1705.05363>`_, Pathak et al, 2017. **算法： Intrinsic Curiosity Module (ICM).**

.. [#] `Large-Scale Study of Curiosity-Driven Learning <https://arxiv.org/abs/1808.04355>`_, Burda et al, 2018. **贡献： 系统分析了基于惊讶度的内在动机方法在多种环境中的表现。

.. [#] `Exploration by Random Network Distillation <https://arxiv.org/abs/1810.12894>`_, Burda et al, 2018. **算法： RND.**


b. 无监督强化学习
-------------------

.. [#] `Variational Intrinsic Control <https://arxiv.org/abs/1611.07507>`_, Gregor et al, 2016. **算法： VIC.**

.. [#] `Diversity is All You Need: Learning Skills without a Reward Function <https://arxiv.org/abs/1802.06070>`_, Eysenbach et al, 2018. **算法： DIAYN.**

.. [#] `Variational Option Discovery Algorithms <https://arxiv.org/abs/1807.10299>`_, Achiam et al, 2018. **算法： VALOR.**


3. 迁移与多任务强化学习
============================

.. [#] `Progressive Neural Networks <https://arxiv.org/abs/1606.04671>`_, Rusu et al, 2016. **算法： Progressive Networks.**

.. [#] `Universal Value Function Approximators <http://proceedings.mlr.press/v37/schaul15.pdf>`_, Schaul et al, 2015. **算法： UVFA.**

.. [#] `Reinforcement Learning with Unsupervised Auxiliary Tasks <https://arxiv.org/abs/1611.05397>`_, Jaderberg et al, 2016. **算法： UNREAL.**

.. [#] `The Intentional Unintentional Agent: Learning to Solve Many Continuous Control Tasks Simultaneously <https://arxiv.org/abs/1707.03300>`_, Cabi et al, 2017. **算法： IU Agent.**

.. [#] `PathNet: Evolution Channels Gradient Descent in Super Neural Networks <https://arxiv.org/abs/1701.08734>`_, Fernando et al, 2017. **算法： PathNet.**

.. [#] `Mutual Alignment Transfer Learning <https://arxiv.org/abs/1707.07907>`_, Wulfmeier et al, 2017. **算法： MATL.**

.. [#] `Learning an Embedding Space for Transferable Robot Skills <https://openreview.net/forum?id=rk07ZXZRb&noteId=rk07ZXZRb>`_, Hausman et al, 2018. 

.. [#] `Hindsight Experience Replay <https://arxiv.org/abs/1707.01495>`_, Andrychowicz et al, 2017. **算法： Hindsight Experience Replay (HER).**

4. 层次化强化学习
===================

.. [#] `Strategic Attentive Writer for Learning Macro-Actions <https://arxiv.org/abs/1606.04695>`_, Vezhnevets et al, 2016. **算法： STRAW.**

.. [#] `FeUdal Networks for Hierarchical Reinforcement Learning <https://arxiv.org/abs/1703.01161>`_, Vezhnevets et al, 2017. **算法： Feudal Networks**

.. [#] `Data-Efficient Hierarchical Reinforcement Learning <https://arxiv.org/abs/1805.08296>`_, Nachum et al, 2018. **算法： HIRO.**

5. 记忆
=========

.. [#] `Model-Free Episodic Control <https://arxiv.org/abs/1606.04460>`_, Blundell et al, 2016. **算法： MFEC.**


.. [#] `Neural Episodic Control <https://arxiv.org/abs/1703.01988>`_, Pritzel et al, 2017. **算法： NEC.**

.. [#] `Neural Map: Structured Memory for Deep Reinforcement Learning <https://arxiv.org/abs/1702.08360>`_, Parisotto and Salakhutdinov, 2017. **算法： Neural Map.**

.. [#] `Unsupervised Predictive Memory in a Goal-Directed Agent <https://arxiv.org/abs/1803.10760>`_, Wayne et al, 2018. **算法： MERLIN.**

.. [#] `Relational Recurrent Neural Networks <https://arxiv.org/abs/1806.01822>`_, Santoro et al, 2018. **算法： RMC.**

6. 基于模型的强化学习
=======================

a. 模型通过学习获得
---------------------

.. [#] `Imagination-Augmented Agents for Deep Reinforcement Learning <https://arxiv.org/abs/1707.06203>`_, Weber et al, 2017. **算法： I2A.**

.. [#] `Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning <https://arxiv.org/abs/1708.02596>`_, Nagabandi et al, 2017. **算法： MBMF.**

.. [#] `Model-Based Value Expansion for Efficient Model-Free Reinforcement Learning <https://arxiv.org/abs/1803.00101>`_, Feinberg et al, 2018. **算法： MVE.**

.. [#] `Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion <https://arxiv.org/abs/1807.01675>`_, Buckman et al, 2018. **算法： STEVE.**

.. [#] `Model-Ensemble Trust-Region Policy Optimization <https://openreview.net/forum?id=SJJinbWRZ&noteId=SJJinbWRZ>`_, Kurutach et al, 2018. **算法： ME-TRPO.**

.. [#] `Model-Based Reinforcement Learning via Meta-Policy Optimization <https://arxiv.org/abs/1809.05214>`_, Clavera et al, 2018. **算法： MB-MPO.**

.. [#] `Recurrent World Models Facilitate Policy Evolution <https://arxiv.org/abs/1809.01999>`_, Ha and Schmidhuber, 2018. 

b. 模型已知
-----------------

.. [#] `Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm <https://arxiv.org/abs/1712.01815>`_, Silver et al, 2017. **算法： AlphaZero.**

.. [#] `Thinking Fast and Slow with Deep Learning and Tree Search <https://arxiv.org/abs/1705.08439>`_, Anthony et al, 2017. **算法： ExIt.**

7. 元强化学习
===============

.. [#] `RL^2: Fast Reinforcement Learning via Slow Reinforcement Learning <https://arxiv.org/abs/1611.02779>`_, Duan et al, 2016. **算法： RL^2.**

.. [#] `Learning to Reinforcement Learn <https://arxiv.org/abs/1611.05763>`_, Wang et al, 2016. 

.. [#] `Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks <https://arxiv.org/abs/1703.03400>`_, Finn et al, 2017. **算法： MAML.**

.. [#] `A Simple Neural Attentive Meta-Learner <https://openreview.net/forum?id=B1DmUzWAW&noteId=B1DmUzWAW>`_, Mishra et al, 2018. **算法： SNAIL.**

8. 强化学习的规模扩展
=======================

.. [#] `Accelerated Methods for Deep Reinforcement Learning <https://arxiv.org/abs/1803.02811>`_, Stooke and Abbeel, 2018. **贡献： 系统分析不同深度强化学习算法中的并行化。 

.. [#] `IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures <https://arxiv.org/abs/1802.01561>`_, Espeholt et al, 2018. **算法： IMPALA.**

.. [#] `Distributed Prioritized Experience Replay <https://openreview.net/forum?id=H1Dy---0Z>`_, Horgan et al, 2018. **算法： Ape-X.**

.. [#] `Recurrent Experience Replay in Distributed Reinforcement Learning <https://openreview.net/forum?id=r1lyTjAqYX>`_, Anonymous, 2018. **算法： R2D2.**

.. [#] `RLlib: Abstractions for Distributed Reinforcement Learning <https://arxiv.org/abs/1712.09381>`_, Liang et al, 2017. **贡献： 一个可扩展的强化学习算法实现库。 `文档链接 <https://ray.readthedocs.io/en/latest/rllib.html>`_


9. 现实世界中的强化学习
=========================

.. [#] `Benchmarking Reinforcement Learning Algorithms on Real-World Robots <https://arxiv.org/abs/1809.07731>`_, Mahmood et al, 2018. 

.. [#] `Learning Dexterous In-Hand Manipulation <https://arxiv.org/abs/1808.00177>`_, OpenAI, 2018. 

.. [#] `QT-Opt: Scalable Deep Reinforcement Learning for Vision-Based Robotic Manipulation <https://arxiv.org/abs/1806.10293>`_, Kalashnikov et al, 2018. **算法： QT-Opt.**

.. [#] `Horizon: Facebook's Open Source Applied Reinforcement Learning Platform <https://arxiv.org/abs/1811.00260>`_, Gauci et al, 2018. 


10. 安全性
============

.. [#] `Concrete Problems in AI Safety <https://arxiv.org/abs/1606.06565>`_, Amodei et al, 2016. **贡献： 建立了安全问题的分类体系，为后续研究提供了重要起点。这些问题需要得到解决！

.. [#] `Deep Reinforcement Learning From Human Preferences <https://arxiv.org/abs/1706.03741>`_, Christiano et al, 2017. **算法： LFP.**

.. [#] `Constrained Policy Optimization <https://arxiv.org/abs/1705.10528>`_, Achiam et al, 2017. **算法： CPO.**

.. [#] `Safe Exploration in Continuous Action Spaces <https://arxiv.org/abs/1801.08757>`_, Dalal et al, 2018. **算法： DDPG+Safety Layer.**

.. [#] `Trial without Error: Towards Safe Reinforcement Learning via Human Intervention <https://arxiv.org/abs/1707.05173>`_, Saunders et al, 2017. **算法： HIRL.**

.. [#] `Leave No Trace: Learning to Reset for Safe and Autonomous Reinforcement Learning <https://arxiv.org/abs/1711.06782>`_, Eysenbach et al, 2017. **算法： Leave No Trace.**


11. 模仿学习与逆强化学习
=========================================================

.. [#] `Modeling Purposeful Adaptive Behavior with the Principle of Maximum Causal Entropy <http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf>`_, Ziebart 2010. **贡献： 对最大熵逆强化学习（IRL）进行了清晰明确的形式化表述。

.. [#] `Guided Cost Learning: Deep Inverse Optimal Control via Policy Optimization <https://arxiv.org/abs/1603.00448>`_, Finn et al, 2016. **算法： GCL.**

.. [#] `Generative Adversarial Imitation Learning <https://arxiv.org/abs/1606.03476>`_, Ho and Ermon, 2016. **算法： GAIL.**

.. [#] `DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills <https://xbpeng.github.io/projects/DeepMimic/2018_TOG_DeepMimic.pdf>`_, Peng et al, 2018. **算法： DeepMimic.**

.. [#] `Variational Discriminator Bottleneck: Improving Imitation Learning, Inverse RL, and GANs by Constraining Information Flow <https://arxiv.org/abs/1810.00821>`_, Peng et al, 2018. **算法： VAIL.**

.. [#] `One-Shot High-Fidelity Imitation: Training Large-Scale Deep Nets with RL <https://arxiv.org/abs/1810.05017>`_, Le Paine et al, 2018. **算法： MetaMimic.**


12. 可复现性、分析与批判
===========================================

.. [#] `Benchmarking Deep Reinforcement Learning for Continuous Control <https://arxiv.org/abs/1604.06778>`_, Duan et al, 2016. **贡献：rllab。**

.. [#] `Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control <https://arxiv.org/abs/1708.04133>`_, Islam et al, 2017.

.. [#] `Deep Reinforcement Learning that Matters <https://arxiv.org/abs/1709.06560>`_, Henderson et al, 2017. 

.. [#] `Where Did My Optimum Go?: An Empirical Analysis of Gradient Descent Optimization in Policy Gradient Methods <https://arxiv.org/abs/1810.02525>`_, Henderson et al, 2018. 

.. [#] `Are Deep Policy Gradient Algorithms Truly Policy Gradient Algorithms? <https://arxiv.org/abs/1811.02553>`_, Ilyas et al, 2018.

.. [#] `Simple Random Search Provides a Competitive Approach to Reinforcement Learning <https://arxiv.org/abs/1803.07055>`_, Mania et al, 2018.

.. [#] `Benchmarking Model-Based Reinforcement Learning <https://arxiv.org/abs/1907.02057>`_, Wang et al, 2019.

13. 补充：经典强化学习理论与综述论文
================================================

.. [#] `Policy Gradient Methods for Reinforcement Learning with Function Approximation <https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf>`_, Sutton et al, 2000. **贡献： 确立了策略梯度定理，并证明了针对任意策略类的策略梯度算法的收敛性。 

.. [#] `An Analysis of Temporal-Difference Learning with Function Approximation <http://web.mit.edu/jnt/www/Papers/J063-97-bvr-td.pdf>`_, Tsitsiklis and Van Roy, 1997. **贡献： 给出了强化学习中价值学习方法的多种收敛性结果和反例。

.. [#] `Reinforcement Learning of Motor Skills with Policy Gradients <http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Neural-Netw-2008-21-682_4867%5b0%5d.pdf>`_, Peters and Schaal, 2008. **贡献： 全面回顾了当时的策略梯度方法，其中许多论述至今仍可用于描述深度强化学习方法。 

.. [#] `Approximately Optimal Approximate Reinforcement Learning <https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf>`_, Kakade and Langford, 2002. **贡献： 包含单调改进理论的早期思想，后来为 TRPO 等算法的理论依据奠定了基础。

.. [#] `A Natural Policy Gradient <https://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf>`_, Kakade, 2002. **贡献： 将自然梯度引入强化学习，随后推动了 TRPO、ACKTR 及深度强化学习中的多种其他方法。

.. [#] `Algorithms for Reinforcement Learning <https://sites.ualberta.ca/~szepesva/papers/RLAlgsInMDPs.pdf>`_, Szepesvari, 2009. **贡献： 一份关于深度强化学习兴起之前的强化学习的极佳参考资料，涵盖基础知识与理论背景。

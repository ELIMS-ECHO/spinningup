---
tags: [SpinningUp, 深度强化学习, 中文教材]
source_rst: docs/spinningup/rl_intro3_zh.rst
source_github: https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/docs/spinningup/rl_intro3_zh.rst
---
# 第三部分：策略优化入门



在本节中，我们将讨论策略优化算法的数学基础，并把相关内容与示例代码联系起来。我们会介绍 **策略梯度（policy gradients）** 理论中的三个重要结果：

- 一个描述策略表现相对于策略参数的梯度的[最简单公式](../spinningup/rl_intro3_zh.md)；
- 一个允许我们从表达式中[去掉无用项](../spinningup/rl_intro3_zh.md) 的规则；
- 一个允许我们向表达式中[加入有用项](../spinningup/rl_intro3_zh.md) 的规则。

最后，我们将这些结果综合起来，介绍基于优势函数的策略梯度表达式。这正是我们在[原始策略梯度（Vanilla Policy Gradient）](../algorithms/vpg_zh.md) 实现中采用的版本。

## 推导最简单的策略梯度

在这里，我们考虑一个随机性的参数化策略 $\pi_{\theta}$。我们希望最大化期望回报 $J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}\left[R(\tau)\right]$。为了推导方便，假设 $R(\tau)$ 表示[有限时域、无折扣回报](../spinningup/rl_intro_zh.md)，不过在无限时域、折扣回报情况下，推导几乎完全相同。

我们希望通过梯度上升来优化策略，例如：

$$\theta_{k+1} = \theta_k + \alpha \left. \nabla_{\theta} J(\pi_{\theta}) \right|_{\theta_k}.$$

策略表现的梯度 $\nabla_{\theta} J(\pi_{\theta})$ 被称为 **策略梯度**，以这种方式优化策略的算法被称为 **策略梯度算法**。（例如原始策略梯度和 TRPO。PPO 也经常被称为策略梯度算法，尽管严格来说这种称呼略有不准确。）

要真正使用这个算法，就需要一个能够进行数值计算的策略梯度表达式。这涉及两个步骤：1）推导策略表现的解析梯度，结果会表现为一个期望；2）使用有限次智能体与环境交互所收集的数据，构造这个期望的样本估计。

本小节将推导最简单的表达式。后续小节会介绍如何改进这个最简单的版本，从而得到标准策略梯度实现实际采用的表达式。

我们先列出一些有助于推导解析梯度的事实。

**1. 一条轨迹的概率。** 如果动作来自 $\pi_{\theta}$，那么轨迹 $\tau = (s_0, a_0, ..., s_{T+1})$ 的概率为：

$$P(\tau|\theta) = \rho_0 (s_0) \prod_{t=0}^{T} P(s_{t+1}|s_t, a_t) \pi_{\theta}(a_t |s_t).$$

**2. 对数求导技巧（Log-Derivative Trick）。** 对数求导技巧基于一条简单的微积分规则： $\log x$ 对 $x$ 的导数为 $1/x$。重新整理并结合链式法则，可以得到：

$$\nabla_{\theta} P(\tau | \theta) = P(\tau | \theta) \nabla_{\theta} \log P(\tau | \theta).$$

**3. 一条轨迹的对数概率。** 一条轨迹的对数概率为：

$$\log P(\tau|\theta) = \log \rho_0 (s_0) + \sum_{t=0}^{T} \bigg( \log P(s_{t+1}|s_t, a_t)  + \log \pi_{\theta}(a_t |s_t)\bigg).$$

**4. 环境函数的梯度。** 环境不依赖 $\theta$，因此 $\rho_0(s_0)$、 $P(s_{t+1}|s_t,a_t)$ 和 $R(\tau)$ 对该参数的梯度都为零。

**5. 轨迹对数概率的梯度。** 因此，轨迹对数概率的梯度为：

$$\begin{aligned}
\nabla_{\theta} \log P(\tau | \theta) &= \cancel{\nabla_{\theta} \log \rho_0 (s_0)} + \sum_{t=0}^{T} \bigg( \cancel{\nabla_{\theta} \log P(s_{t+1}|s_t, a_t)}  + \nabla_{\theta} \log \pi_{\theta}(a_t |s_t)\bigg) \\
&= \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t).
\end{aligned}$$

把这些结果结合起来，可以推导得到：

> [!note] 基础策略梯度的推导
> <span nowrap="">$$\begin{aligned}
> \begin{align*}
> \nabla_{\theta} J(\pi_{\theta}) &= \nabla_{\theta} \mathbb{E}_{\tau \sim \pi_{\theta}}\left[R(\tau)\right] & \\
> &= \nabla_{\theta} \int_{\tau} P(\tau|\theta) R(\tau) & \text{Expand expectation} \\
> &= \int_{\tau} \nabla_{\theta} P(\tau|\theta) R(\tau) & \text{Bring gradient under integral} \\
> &= \int_{\tau} P(\tau|\theta) \nabla_{\theta} \log P(\tau|\theta) R(\tau) & \text{Log-derivative trick} \\
> &= \mathbb{E}_{\tau \sim \pi_{\theta}}\left[\nabla_{\theta} \log P(\tau|\theta) R(\tau)\right] & \text{Return to expectation form} \\
> \therefore \nabla_{\theta} J(\pi_{\theta}) &= \mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)\right] & \text{Expression for grad-log-prob}
> \end{align*}
> \end{aligned}$$</span>


这是一个期望值，因此我们可以使用样本均值来估计。如果收集了一组轨迹 $\mathcal{D} = \{\tau_i\}_{i=1,...,N}$，每条轨迹都是智能体按照策略 $\pi_{\theta}$ 在环境中行动得到的，那么策略梯度就可以估计为：

$$\hat{g} = \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau),$$

其中 $|\mathcal{D}|$ 是 $\mathcal{D}$ 中的轨迹数量（即这里的 $N$）。

最后这个表达式就是我们希望得到的可计算表达式的最简单版本。假设策略的表示方式允许计算 $\nabla_{\theta} \log \pi_{\theta}(a|s)$，而且能够在环境中运行该策略以收集轨迹数据集，我们就可以计算策略梯度，并进行一次更新。

## 实现最简单的策略梯度

我们在 <span class="title-ref">spinup/examples/pytorch/pg_math/1_simple_pg.py</span> 中提供了这一简单策略梯度算法的简短 PyTorch 实现。（也可以[在 GitHub 上查看](https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py)。）整个文件只有 128 行，因此我们强烈推荐你深入阅读。这里不会介绍所有代码，但会重点说明并解释几个重要部分。

> [!note] 你应该知道
> 本节以前使用 TensorFlow 示例编写。旧的 TensorFlow 内容可以在[这里](https://github.com/openai/spinningup/blob/master/docs/spinningup/extra_tf_pg_implementation.rst#implementing-the-simplest-policy-gradient) 找到。


**1. 构建策略网络。**

``` python
# make core of policy network
logits_net = mlp(sizes=[obs_dim]+hidden_sizes+[n_acts])

# make function to compute action distribution
def get_policy(obs):
    logits = logits_net(obs)
    return Categorical(logits=logits)

# make action selection function (outputs int actions, sampled from policy)
def get_action(obs):
    return get_policy(obs).sample().item()
```

这一代码块建立了一些模块和函数，用来使用前馈神经网络构造一个分类策略。（如果需要复习，可以参考第一部分中的[随机性策略](../spinningup/rl_intro_zh.md)。）\`logits_net\` 模块的输出可用于构造动作的对数概率与概率，而 <span class="title-ref">get_action</span> 函数根据 logits 计算出的概率采样动作。（注意：这个特定的 <span class="title-ref">get_action</span> 函数假设仅提供一个 <span class="title-ref">obs</span>，因此只输出一个整数动作。这就是它使用 <span class="title-ref">.item()</span> 的原因，该方法用于[取得只有一个元素的 Tensor 的内容](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.item)。）

这里的大量工作由第 36 行的 <span class="title-ref">Categorical</span> 对象完成。它是 PyTorch 的一个 <span class="title-ref">Distribution</span> 对象，封装了一些与概率分布有关的数学函数。具体来说，它提供了从分布中采样的方法（第 40 行使用），以及计算给定样本的对数概率的方法（后面会用到）。PyTorch 的分布对象在强化学习中很有用，建议阅读[其文档](https://pytorch.org/docs/stable/distributions.html) 来了解其工作方式。

> [!note] 你应该知道
> 再提醒一次：当我们说一个分类分布具有“logits”时，意思是每个结果的概率由 logits 的 Softmax 函数计算。也就是说，对一个 logits 为 $x_j$ 的分类分布，动作 $j$ 的概率为：
>
> $$p_j = \frac{\exp(x_j)}{\sum_{i} \exp(x_i)}$$


**2. 构建损失函数。**

``` python
# make loss function whose gradient, for the right data, is policy gradient
def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()
```

这段代码为策略梯度算法构造了一个“损失”函数。当输入合适的数据时，这个损失的梯度等于策略梯度。合适的数据，是指在按照当前策略行动时收集的一组（状态、动作、权重）三元组，其中一个状态—动作对的权重，是它所在回合的回报。（不过，后面的小节会说明，还可以选择一些其他权重，并且同样正确。）

> [!note] 你应该知道
> 虽然我们将它称为损失函数，但它 **不是** 监督学习中通常意义的损失函数。它与标准损失函数有两个主要区别。
>
> **1. 数据分布依赖于参数。** 损失函数通常定义在一个固定的数据分布上，这个分布独立于要优化的参数。而这里并非如此，因为数据必须从最新的策略中采样。
>
> **2. 它不衡量性能。** 损失函数通常评价我们关心的性能指标。这里我们关心的是期望回报 $J(\pi_{\theta})$，但我们的“损失”函数完全不能近似它，即使从期望意义上说也是如此。这个“损失”函数对我们有用，只是因为：当在当前参数位置对其求值，且所用数据也是由当前参数生成时，它具有性能梯度的负值。
>
> 但在第一次梯度下降更新之后，它就不再与性能有这样的联系。这意味着，对给定的一批数据进行这个“损失”函数的最小化，\**绝对不能保证*\* 期望回报得到改善。你可以让这个损失下降到 $-\infty$，同时策略性能可能一落千丈；实际上通常确实如此。深度强化学习研究者有时会把这一结果描述为策略对一批数据“过拟合”。这只是描述性的说法，不应按字面理解，因为它并不是指泛化误差。
>
> 我们强调这一点，是因为机器学习实践者经常把损失函数视为训练中有意义的信号，认为“只要损失下降，就说明一切正常”。在策略梯度中，这种直觉是错误的，你应该只关心平均回报。损失函数本身没有意义。


> [!note] 你应该知道
> 这里通过调用 PyTorch <span class="title-ref">Categorical</span> 对象的 <span class="title-ref">log_prob</span> 方法来生成 <span class="title-ref">logp</span> 张量。对于其他类型的分布对象，这种方法可能需要修改。
>
> 例如，如果使用[正态分布](https://pytorch.org/docs/stable/distributions.html#normal)（对应对角高斯策略），调用 <span class="title-ref">policy.log_prob(act)</span> 会返回一个 Tensor，其中包含每个向量动作每个分量的单独对数概率。也就是说，当你输入一个形状为 <span class="title-ref">(batch, act_dim)</span> 的 Tensor，会得到一个形状同样为 <span class="title-ref">(batch, act_dim)</span> 的 Tensor，但为了构建强化学习的损失函数，你需要的是形状为 <span class="title-ref">(batch,)</span> 的 Tensor。因此，你需要把动作各分量的对数概率加起来，以得到整个动作的对数概率。即：
>
> ``` python
> logp = get_policy(obs).log_prob(act).sum(axis=-1)
> ```


**3. 运行一个训练轮次。**

``` python
# for training policy
def train_one_epoch():
    # make some empty lists for logging.
    batch_obs = []          # for observations
    batch_acts = []         # for actions
    batch_weights = []      # for R(tau) weighting in policy gradient
    batch_rets = []         # for measuring episode returns
    batch_lens = []         # for measuring episode lengths

    # reset episode-specific variables
    obs = env.reset()       # first obs comes from starting distribution
    done = False            # signal from environment that episode is over
    ep_rews = []            # list for rewards accrued throughout ep

    # render first episode of each epoch
    finished_rendering_this_epoch = False

    # collect experience by acting in the environment with current policy
    while True:

        # rendering
        if (not finished_rendering_this_epoch) and render:
            env.render()

        # save obs
        batch_obs.append(obs.copy())

        # act in the environment
        act = get_action(torch.as_tensor(obs, dtype=torch.float32))
        obs, rew, done, _ = env.step(act)

        # save action, reward
        batch_acts.append(act)
        ep_rews.append(rew)

        if done:
            # if episode is over, record info about episode
            ep_ret, ep_len = sum(ep_rews), len(ep_rews)
            batch_rets.append(ep_ret)
            batch_lens.append(ep_len)

            # the weight for each logprob(a|s) is R(tau)
            batch_weights += [ep_ret] * ep_len

            # reset episode-specific variables
            obs, done, ep_rews = env.reset(), False, []

            # won't render again this epoch
            finished_rendering_this_epoch = True

            # end experience loop if we have enough of it
            if len(batch_obs) > batch_size:
                break

    # take a single policy gradient update step
    optimizer.zero_grad()
    batch_loss = compute_loss(obs=torch.as_tensor(batch_obs, dtype=torch.float32),
                              act=torch.as_tensor(batch_acts, dtype=torch.int32),
                              weights=torch.as_tensor(batch_weights, dtype=torch.float32)
                              )
    batch_loss.backward()
    optimizer.step()
    return batch_loss, batch_rets, batch_lens
```

<span class="title-ref">train_one_epoch()</span> 函数运行一个策略梯度训练“轮次”（epoch），在这里我们将其定义为：

1）经验收集步骤（第 67—102 行），智能体使用最新策略在环境中运行若干回合；

2）随后执行一次策略梯度更新（第 104—111 行）。

算法的主循环只是反复调用 <span class="title-ref">train_one_epoch()</span>。

> [!note] 你应该知道
> 如果你还不熟悉 PyTorch 中的优化过程，请观察第 104—111 行所展示的一次梯度下降更新模式。首先，清空梯度缓冲区。然后计算损失函数。接着对损失执行反向传播，这会把新计算出的梯度累积到梯度缓冲区。最后，让优化器执行一步更新。


## 期望对数概率梯度引理

本小节将推导一个在策略梯度理论中得到广泛使用的中间结果。我们把它称为“期望对数概率梯度”（Expected Grad-Log-Prob，EGLP）引理。

**EGLP 引理。** 假设 $P_{\theta}$ 是随机变量 $x$ 上的一个参数化概率分布，那么：

$$\mathbb{E}_{x \sim P_{\theta}}\left[\nabla_{\theta} \log P_{\theta}(x)\right] = 0.$$

> [!note] 证明
> 回想一下，所有概率分布都是 *归一化* 的：
>
> $$\int_x P_{\theta}(x) = 1.$$
>
> 对归一化条件两边取梯度：
>
> $$\nabla_{\theta} \int_x P_{\theta}(x) = \nabla_{\theta} 1 = 0.$$
>
> 使用对数求导技巧，得到：
>
> $$\begin{aligned}
> 0 &= \nabla_{\theta} \int_x P_{\theta}(x) \\
> &= \int_x \nabla_{\theta} P_{\theta}(x) \\
> &= \int_x P_{\theta}(x) \nabla_{\theta} \log P_{\theta}(x) \\
> \therefore 0 &= \mathbb{E}_{x \sim P_{\theta}}\left[\nabla_{\theta} \log P_{\theta}(x)\right].
> \end{aligned}$$


- 文献 1： 本文作者还不知道文献中有任何地方给这个引理赋予了一个标准名称。不过，鉴于它出现得如此频繁，为方便引用，似乎很值得给它起个名字。

## 不要让过去分散你的注意力

观察我们最近得到的策略梯度表达式：

$$\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) R(\tau)\right].$$

沿着这个梯度走一步，会按照 $R(\tau)$（也就是 *全部时间内获得的奖励之和*）的大小，提高每个动作的对数概率。但这样做不太合理。

智能体其实应该只根据动作的 *后果* 来强化动作。一个动作执行之前获得的奖励，并不能说明这个动作是否好；真正有关的是动作 *之后* 的奖励。

事实证明，这一直觉也反映在数学中，我们可以证明策略梯度还可以表达为：

$$\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1})\right].$$

在这种表达方式下，只会根据动作执行以后获得的奖励来强化动作。

我们把这种形式称为“reward-to-go 策略梯度”，因为轨迹中某一点之后的奖励之和：

$$\hat{R}_t \doteq \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}),$$

被称为该点的 **未来回报（reward-to-go）**，而这个策略梯度表达式依赖各个状态—动作对对应的未来回报。

> [!note] 你应该知道
> **但这样为什么更好？** 策略梯度的一个关键问题在于，需要多少条样本轨迹才能得到低方差的梯度样本估计。我们最初使用的公式包含了一些根据过去奖励强化动作的项。这些项的均值为零，但方差不为零，因此它们只会给策略梯度的样本估计增加噪声。把它们去掉，可以减少需要采样的轨迹数量。


这一论断的（可选）证明可以在[这里](https://github.com/openai/spinningup/blob/master/docs/spinningup/extra_pg_proof1.rst) 找到，而该证明最终依赖于 EGLP 引理。

## 实现 Reward-to-Go 策略梯度

我们在 <span class="title-ref">spinup/examples/pytorch/pg_math/2_rtg_pg.py</span> 中给出了 reward-to-go 策略梯度的简短 PyTorch 实现。（也可以[在 GitHub 上查看](https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/2_rtg_pg.py)。）

与 <span class="title-ref">1_simple_pg.py</span> 相比，唯一改变的是损失函数所使用的权重。代码修改很小：我们增加一个新函数，并改动另外两行。新函数如下：

``` python
def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs
```

然后，我们把原先第 91—92 行的代码：

``` python
# the weight for each logprob(a|s) is R(tau)
batch_weights += [ep_ret] * ep_len
```

改成：

``` python
# the weight for each logprob(a_t|s_t) is reward-to-go from t
batch_weights += list(reward_to_go(ep_rews))
```

> [!note] 你应该知道
> 本节以前使用 TensorFlow 示例编写。旧的 TensorFlow 内容可以在[这里](https://github.com/openai/spinningup/blob/master/docs/spinningup/extra_tf_pg_implementation.rst#implementing-reward-to-go-policy-gradient) 找到。


## 策略梯度中的基线

EGLP 引理的一个直接推论是，对任何只依赖于状态的函数 $b$：

$$\mathbb{E}_{a_t \sim \pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(a_t|s_t) b(s_t)\right] = 0.$$

因此，我们可以向策略梯度表达式中添加任意数量的这类项，或从中减去这类项，而不会改变它的期望：

$$\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \left(\sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t)\right)\right].$$

以这种方式使用的函数 $b$，称为 **基线（baseline）**。

最常见的基线选择是[给定策略下的状态价值函数](../spinningup/rl_intro_zh.md) $V^{\pi}(s_t)$。回想一下，它表示智能体从状态 $s_t$ 开始，并在之后始终按照策略 $\pi$ 行动，所能获得的平均回报。

根据经验，选择 $b(s_t) = V^{\pi}(s_t)$ 可以降低策略梯度样本估计的方差。这使得策略学习更快、更稳定。从概念上看，这种选择也很有吸引力：它体现了一种直觉，即如果智能体获得了自己预期的结果，那么它应对此“没有特别的感觉”。

> [!note] 你应该知道
> 实践中，我们无法精确计算 $V^{\pi}(s_t)$，因此必须近似它。通常使用神经网络 $V_{\phi}(s_t)$，并在更新策略的同时更新这个网络（使价值网络始终近似最新策略的价值函数）。
>
> 学习 $V_{\phi}$ 最简单的方式，是最小化一个均方误差目标。大多数策略优化算法的实现（包括 VPG、TRPO、PPO 和 A2C）都采用它：
>
> $$\phi_k = \arg \min_{\phi} \mathbb{E}_{s_t, \hat{R}_t \sim \pi_k}\left[\left( V_{\phi}(s_t) - \hat{R}_t \right)^2\right],$$
>
> 其中 $\pi_k$ 是第 $k$ 个训练轮次的策略。具体做法是从之前的价值参数 $\phi_{k-1}$ 开始，执行一次或多次梯度下降。


## 策略梯度的其他形式

到目前为止，我们看到策略梯度的一般形式为：

$$\nabla_{\theta} J(\pi_{\theta}) = \mathbb{E}_{\tau \sim \pi_{\theta}}\left[\sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t |s_t) \Phi_t\right],$$

其中 $\Phi_t$ 可以采用以下任意一种形式：

$$\Phi_t &= R(\tau),$$

或者：

$$\Phi_t &= \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}),$$

或者：

$$\Phi_t &= \sum_{t'=t}^T R(s_{t'}, a_{t'}, s_{t'+1}) - b(s_t).$$

尽管这些选择具有不同的方差，但它们得到的策略梯度期望值相同。事实上，还有两种同样有效的 $\Phi_t$ 权重值得了解。

**1. 给定策略下的动作价值函数（On-Policy Action-Value Function）。** 选择：

$$\Phi_t = Q^{\pi_{\theta}}(s_t, a_t)$$

也是有效的。可参阅[这一页](https://github.com/openai/spinningup/blob/master/docs/spinningup/extra_pg_proof2.rst)，查看该结论的（可选）证明。

**2. 优势函数（Advantage Function）。** 回想一下，一个动作的[优势](../spinningup/rl_intro_zh.md) 定义为 $A^{\pi}(s_t,a_t) = Q^{\pi}(s_t,a_t) - V^{\pi}(s_t)$，表示它与当前策略下其他动作相比，平均而言有多好或多差。选择：

$$\Phi_t = A^{\pi_{\theta}}(s_t, a_t)$$

同样有效。这是因为它等价于选择 $\Phi_t = Q^{\pi_{\theta}}(s_t, a_t)$，然后使用一个价值函数基线，而我们始终可以这么做。

> [!note] 你应该知道
> 用优势函数表示策略梯度是极其常见的做法，不同算法也采用了许多不同的优势函数估计方法。


> [!note] 你应该知道
> 如果希望更深入地了解这个主题，应该阅读[广义优势估计（Generalized Advantage Estimation）](https://arxiv.org/abs/1506.02438)（GAE）论文。该论文在背景部分深入讨论了 $\Phi_t$ 的不同选择。
>
> 随后，这篇论文介绍了 GAE：一种在策略优化算法中近似优势函数的方法，它的使用非常广泛。例如，Spinning Up 对 VPG、TRPO 和 PPO 的实现就采用了它。因此，我们强烈建议你学习这一方法。


## 回顾

本章介绍了策略梯度方法的基础理论，并把一些早期结果与代码示例联系起来。感兴趣的学习者可以继续研究后面的结果（价值函数基线与使用优势函数表示策略梯度），如何转化为 Spinning Up 对[原始策略梯度（Vanilla Policy Gradient）](../algorithms/vpg_zh.md) 的实现。

---
tags: [SpinningUp, 深度强化学习, 中文教材]
source_rst: docs/user/installation_zh.rst
source_github: https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/docs/user/installation_zh.rst
---
# 安装



Spinning Up 需要 Python 3、OpenAI Gym 和 OpenMPI。

Spinning Up 目前仅支持 Linux 和 OS X。虽然也可能在 Windows 上安装，但尚未经过充分测试。[^1]

> [!note] 你应该知道
> Spinning Up 的许多示例和基准测试使用 [MuJoCo](http://www.mujoco.org/index.html) 物理引擎。原文撰写时，MuJoCo 是需要许可证的专有软件，提供免费试用及学生免费许可，其他情况下则需要付费。因此安装是可选的；不过，由于它实际上是连续控制深度强化学习基准测试的通行标准，作者仍建议安装。
>
> 即使不安装 MuJoCo，也完全可以在 Gym 免费提供的 [Classic Control](https://gym.openai.com/envs/#classic_control) 和 [Box2d](https://gym.openai.com/envs/#box2d) 环境中运行强化学习算法，开始学习。


## 安装 Python

我们建议通过 Anaconda 安装 Python。Anaconda 包含 Python、许多实用的软件包，以及名为 conda 的环境管理器，能够简化软件包管理。

按照 [Anaconda 安装说明](https://docs.continuum.io/anaconda/install/) 下载并安装 Anaconda3（原文撰写时的版本为 [Anaconda3-5.3.0](https://repo.anaconda.com/archive/)）。然后创建使用 Python 3.6 的 conda 环境，以管理 Spinning Up 所需的软件包：

<div class="parsed-literal">

conda create -n spinningup python=3.6

</div>

要使用刚创建的环境中的 Python，请激活环境：

<div class="parsed-literal">

conda activate spinningup

</div>

> [!note] 你应该知道
> 如果刚接触 Python 环境和软件包管理，这些内容很容易令人困惑，也可能遇到一些问题（尤其是“我明明安装了这个东西，但使用时却提示找不到！”）。建议阅读关于软件包管理是什么、为什么有用，以及如何正确使用常见命令的清晰介绍。
>
> [FreeCodeCamp](https://medium.freecodecamp.org/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c) 有一篇值得阅读的解释；[Towards Data Science](https://towardsdatascience.com/environment-management-with-conda-python-2-3-b9961a8a5097) 上还有一篇更简短且实用的介绍。如果你非常有耐心，也可以阅读 [Conda 的文档页面](https://conda.io/docs/user-guide/tasks/manage-environments.html)，虽然枯燥，但内容很丰富。


## 安装 OpenMPI

### Ubuntu

<div class="parsed-literal">

sudo apt-get update && sudo apt-get install libopenmpi-dev

</div>

### Mac OS X

在 Mac 上安装系统软件包需要 [Homebrew](https://brew.sh)。安装 Homebrew 后，运行以下命令：

<div class="parsed-literal">

brew install openmpi

</div>

## 安装 Spinning Up

<div class="parsed-literal">

git clone <https://github.com/openai/spinningup.git> cd spinningup pip install -e .

</div>

> [!note] 你应该知道
> Spinning Up 默认安装 Gym 中除 MuJoCo 环境\**以外*\*的全部内容。如果安装 Gym 时遇到问题，可以查看 [Gym](https://github.com/openai/gym) 的 GitHub 页面。如果希望使用 MuJoCo 环境，请参阅下文的可选安装章节。


## 检查安装结果

要检查是否成功安装 Spinning Up，可以尝试在 LunarLander-v2 环境中运行 PPO：

<div class="parsed-literal">

python -m spinup.run ppo --hid "\[32,32\]" --env LunarLander-v2 --exp_name installtest --gamma 0.999

</div>

这一过程可能需要约 10 分钟。你可以让它在后台运行，同时继续阅读文档。这不会将智能体训练至完全收敛，但足以在结果中看到\*一定程度的\*学习进展。

训练完成后，使用以下命令观看训练后策略的视频：

<div class="parsed-literal">

python -m spinup.run test_policy data/installtest/installtest_s0

</div>

使用以下命令绘制结果：

<div class="parsed-literal">

python -m spinup.run plot data/installtest/installtest_s0

</div>

## 安装 MuJoCo（可选）

首先，前往 [mujoco-py](https://github.com/openai/mujoco-py) 的 GitHub 页面。按照 README 中的说明，安装 MuJoCo 物理引擎和 mujoco-py 软件包（后者让 Python 能够使用 MuJoCo）。

> [!note] 你应该知道
> 按照原文撰写时的要求，使用 MuJoCo 模拟器需要获取 [MuJoCo 许可证](https://www.roboti.us/license.html)。任何人都可获得免费的 30 天许可证，全日制学生可获得免费的 1 年许可证。


安装 MuJoCo 后，使用以下命令安装对应的 Gym 环境：

<div class="parsed-literal">

pip install gym\[mujoco,robotics\]

</div>

然后在 Walker2d-v2 环境中运行 PPO，检查是否正常工作：

<div class="parsed-literal">

python -m spinup.run ppo --hid "\[32,32\]" --env Walker2d-v2 --exp_name mujocotest

</div>

[^1]: 看起来至少有一位用户找到了 [在 Windows 上运行的变通方法](https://github.com/openai/spinningup/issues/23)。如果你通过其他方法成功运行，也欢迎告诉作者具体做法。

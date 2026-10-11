---
tags: [SpinningUp, 深度强化学习, 中文教材]
source_rst: docs/user/running_zh.rst
source_github: https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/docs/user/running_zh.rst
---
# 运行实验



熟悉深度强化学习的最佳方式之一，就是运行算法，观察它们在不同任务上的表现。Spinning Up 代码库让小规模（本地）实验变得容易。本节介绍两种运行方式：从命令行启动，或通过脚本中的函数调用启动。

## 从命令行启动

Spinning Up 附带 `spinup/run.py`，它是一个方便的工具，可让你从命令行轻松启动任意算法，并选择任意超参数。它也对查看训练后策略和绘图的工具提供轻量级封装，不过本页不讨论这些功能（详见 [实验输出](../user/saving_and_loading_zh.md) 和 [绘图](../user/plotting_zh.md) 页面）。

从命令行运行 Spinning Up 算法的标准方式是：


```text
python -m spinup.run \[algo name\] \[experiment flags\]
```


例如：


```text
python -m spinup.run ppo --env Walker2d-v2 --exp_name walker
```


> [!note] 你应该知道
> 如果使用 ZShell：ZShell 会把方括号解释为特殊字符。Spinning Up 的一些命令行参数会使用方括号，因此请确保对它们进行转义。如果希望默认转义，也可以尝试 [这里推荐的解决方法](http://kinopyo.com/en/blog/escape-square-bracket-by-default-in-zsh)。


> [!note] 详细快速入门指南
> 
```text
>
> python -m spinup.run ppo --exp_name ppo_ant --env Ant-v2 --clip_ratio 0.1 0.2  
> --hid\[h\] \[32,32\] \[64,32\] --act torch.nn.Tanh --seed 0 10 20 --dt --data_dir path/to/data


这条命令在 Gym 的 `Ant-v2` 环境中运行 PPO，并通过各项参数控制不同设置。

默认运行 PyTorch 版本（TRPO 除外，因为 Spinning Up 尚未提供 PyTorch 版 TRPO）。若要运行 TensorFlow 版本，将 `ppo` 替换为 `ppo_tf1`。

`clip_ratio`、`hid` 和 `act` 用于设置算法的部分超参数。可以为超参数提供多个值，从而运行多个实验。要了解可以设置哪些超参数，请查阅相应的算法文档（点击这里查看 [PPO 文档](../algorithms/ppo_zh.md)）。

`hid` 和 `act` 是设置神经网络隐藏层大小和激活函数的 [特殊快捷参数](../user/running_zh.md)。

`seed` 参数设置随机数生成器的种子。强化学习算法的结果方差较大，因此应尝试多个种子，以了解性能的变化情况。

`dt` 参数确保保存目录名称包含时间戳（否则不会包含，除非在 `spinup/user_config.py` 中设置 `FORCE_DATESTAMP=True`）。

`data_dir` 参数用于设置结果保存文件夹。默认值由 `spinup/user_config.py` 中的 `DEFAULT_DATA_DIR` 指定；除非修改，否则它是 `spinningup` 文件夹下的 `data` 子文件夹。

[保存目录名称](../user/running_zh.md) 根据 `exp_name` 及具有多个取值的参数决定。目录名中会使用参数的缩写，而不是完整参数名。用户可以在参数后的方括号中指定缩写，例如 `--hid[h]`；否则会自动使用参数名称的子串（例如 `clip_ratio` 变成 `cli`）。例如，当 `clip_ratio=0.1`、`hid=[32,32]`、`seed=10` 时，保存目录为：

<div class="parsed-literal">

path/to/data/YY-MM-DD_ppo_ant_cli0-1_h32-32/YY-MM-DD_HH-MM-SS-ppo_ant_cli0-1_h32-32_seed10
```


</div>

### 从命令行选择 PyTorch 或 TensorFlow

使用算法的 PyTorch 版本时，运行：


```text
python -m spinup.run [\[algo\]]()pytorch
```


使用 TensorFlow 版本时，运行：


```text
python -m spinup.run [\[algo\]]()tf1
```


如果运行 `python -m spinup.run [algo]`，不添加 `_pytorch` 或 `_tf1`，运行器就会查看 `spinup/user_config.py`，确定该算法应使用的默认版本。

### 从命令行设置超参数

所有算法的每个超参数都可以直接从命令行控制。如果 `kwarg` 是算法函数的有效关键字参数，就可以使用 `--kwarg` 标志为它设置值。要查看可用的关键字参数，可以阅读算法文档，或者尝试：


```text
python -m spinup.run \[algo name\] --help
```


以查看函数文档字符串。

> [!note] 你应该知道
> 参数值在使用前会经过 `eval()` 处理，因此可以直接从命令行描述某些函数和对象。例如：
>
> 
```text
>
> python -m spinup.run ppo --env Walker2d-v2 --exp_name walker --act torch.nn.ELU


这会将 `torch.nn.ELU` 设置为激活函数。（TensorFlow 中对应的方式是运行 `ppo_tf1` 并传入 `--act tf.nn.elu`。）
```


> [!note] 你应该知道
> 对于接收字典值的关键字参数，提供了方便的处理方式。不必写：
>
> 
```text
>
> --key dict(v1=value_1, v2=value_2)


可以改写为：

<div class="parsed-literal">

--key:v1 value_1 --key:v2 value_2
```


两者效果相同。

</div>

### 一次启动多个实验

只需为某个参数提供多个值，就可以启动多个实验，并\**依次执行*\*。（每一种可能的参数取值组合都会启动一个实验。）

例如，要使用三个不同的随机种子（0、10、20）运行其他设置相同的实验，可以执行：


```text
python -m spinup.run ppo --env Walker2d-v2 --exp_name walker --seed 0 10 20
```


实验不会并行启动，因为它们会占用大量资源，同时运行多个实验并不能带来加速。

### 特殊参数

少数参数会受到特殊处理。

#### 环境参数

<div class="option">

--env, --env_name

*字符串*。OpenAI Gym 中的环境名称。所有 Spinning Up 算法都实现为接收 `env_fn` 参数的函数，`env_fn` 必须是一个可调用函数，用于创建强化学习环境的副本。由于最常见的用例是通过 `gym.make(env_name)` 创建 Gym 环境，因此命令行允许直接指定 `env_name`（或简写为 `env`），它会被转换为创建对应 Gym 环境的 lambda 函数。

</div>

#### 快捷参数

部分算法参数名称比较长，因此提供了快捷写法：

<div class="option">

--hid, --ac_kwargs:hidden_sizes

*整数列表*。设置神经网络（策略网络和价值函数网络）隐藏层的大小。

</div>

<div class="option">

--act, --ac_kwargs:activation

*TensorFlow 操作*。Actor 和 Critic 神经网络的激活函数。

</div>

这些参数对 Spinning Up 当前的所有算法均有效。

#### 配置参数

这些参数不是任何算法的超参数，而是改变实验配置。

<div class="option">

--cpu, --num_cpu

*整数*。设置后，实验会使用指定数量的进程启动，每个 CPU 对应一个进程，并通过 MPI 连接。某些算法适合这种并行化方式，但并非所有算法都适合。如果对不兼容的算法设置 `num_cpu` 大于 1，就会报错。也可以设置 `--num_cpu auto`，自动使用机器上所有可用 CPU。

</div>

<div class="option">

--exp_name

*字符串*。实验名称，用于命名每次实验的保存目录。默认值为“cmd”加上算法名称。

</div>

<div class="option">

--data_dir

*路径*。设置本次实验或一组实验的基础保存目录。如果没有指定，则使用 `spinup/user_config.py` 中的 `DEFAULT_DATA_DIR`。

</div>

<div class="option">

--datestamp

*布尔值*。在实验保存目录的名称中包含日期和时间。

</div>

### 结果保存在哪里

某个实验（即一种超参数配置的一次运行）的结果保存在：

    data_dir/[outer_prefix]exp_name[suffix]/[inner_prefix]exp_name[suffix]_s[seed]

其中：

- `data_dir` 是 `--data_dir` 参数的值；未指定时，默认为 `spinup/user_config.py` 中的 `DEFAULT_DATA_DIR`。
- 如果启用 `--datestamp`，`outer_prefix` 为 `YY-MM-DD_` 时间戳，否则为空。
- 如果启用 `--datestamp`，`inner_prefix` 为 `YY-MM-DD_HH-MM-SS-` 时间戳，否则为空。
- `suffix` 是根据实验超参数确定的特殊字符串。

#### 后缀如何确定？

只有一次运行多个实验时才会添加后缀。后缀仅包含实验之间取值不同的超参数，随机种子除外。这样做是为了让相似实验（除随机种子外其他参数都相同）归入同一个文件夹。

后缀由超参数的\*缩写\*和对应取值组合而成。缩写可以根据参数名自动构造，也可以由用户提供。用户可以在关键字参数标志后使用方括号指定缩写。

例如：


```text
python -m spinup.run ddpg_tf1 --env Hopper-v2 --hid\[h\] \[300\] \[128,128\] --act tf.nn.tanh tf.nn.relu
```


这里，`--hid` 使用了\**用户提供的缩写*\* `h`。`--act` 没有用户提供的缩写，因此系统会自动生成。

本例产生的后缀为：


```text
\_h128-128_ac-actrelu \_h128-128_ac-acttanh \_h300_ac-actrelu \_h300_ac-acttanh
```


注意，`h` 是用户提供的；`ac-act` 是根据 `ac_kwargs:activation`（`act` 参数的实际名称）自动生成的。

### 补充说明

> [!note] 这一点其实不需要了解
> 每个算法都位于 `spinup/algos/BACKEND/ALGO_NAME/ALGO_NAME.py` 文件中，也可以直接从命令行运行这些文件，但只能使用有限的一组参数，其中部分参数与 `spinup/run.py` 支持的不同。这种单个算法文件中的命令行支持基本属于历史遗留功能，\**不推荐*\* 以此开展实验。
>
> 本文不会介绍这些命令行调用，\*只会\* 介绍通过 `spinup/run.py` 运行的方式。


## 从脚本启动

每个算法都实现为 Python 函数，可以直接从 `spinup` 软件包导入，例如：

``` text
>>> from spinup import ppo_pytorch as ppo
```

每个算法可接受的完整参数列表，请参阅对应的算法文档。可以使用这些方法设置专门的自定义实验，例如：

``` python
from spinup import ppo_tf1 as ppo
import tensorflow as tf
import gym

env_fn = lambda : gym.make('LunarLander-v2')

ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu)

logger_kwargs = dict(output_dir='path/to/output_dir', exp_name='experiment_name')

ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=250, logger_kwargs=logger_kwargs)
```

### 使用 ExperimentGrid

在机器学习研究中，使用多种可能的超参数运行同一算法通常很有用。Spinning Up 提供了一个简单工具来辅助完成这项工作，名为 [ExperimentGrid](../utils/run_utils_zh.md)。

请看 `spinup/examples/pytorch/bench_ppo_cartpole.py` 中的示例：

``` python
from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=3)
    args = parser.parse_args()

    eg = ExperimentGrid(name='ppo-pyt-bench')
    eg.add('env_name', 'CartPole-v0', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 10)
    eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', [(32,), (64,64)], 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.Tanh, torch.nn.ReLU], '')
    eg.run(ppo_pytorch, num_cpu=args.cpu)
```

（等价的 TensorFlow 示例位于 `spinup/examples/tf1/bench_ppo_cartpole.py`。）

创建 ExperimentGrid 对象后，使用以下方式添加参数：


```text
eg.add(param_name, values, shorthand, in_name)
```


其中，`in_name` 会强制某个参数出现在实验名称中，即使它在所有实验中的值都相同。

添加全部参数后，运行：


```text
eg.run(thunk, \*\*run_kwargs)
```


即可运行网格中的所有实验（每种有效配置运行一次），将配置作为关键字参数传给函数 `thunk`。`ExperimentGrid.run` 使用名为 [call_experiment](../utils/run_utils_zh.md) 的函数启动 `thunk`，而 `**run_kwargs` 则指定 `call_experiment` 的行为。详见 [文档页面](../utils/run_utils_zh.md)。

除了不能使用快捷关键字参数（在 `ExperimentGrid` 中不能用 `hid` 代替 `ac_kwargs:hidden_sizes`）之外，`ExperimentGrid` 的基本行为与命令行运行方式相同。（实际上，`spinup.run` 内部就使用了 `ExperimentGrid`。）

---
tags: [SpinningUp, 深度强化学习, 中文教材]
source_rst: docs/user/saving_and_loading_zh.rst
source_github: https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/docs/user/saving_and_loading_zh.rst
---
# 实验输出



本节将介绍：

- Spinning Up 各算法实现会产生哪些输出，
- 这些输出以什么格式存储，以及如何组织，
- 它们保存在哪里，以及如何修改保存位置，
- 如何加载并运行训练好的策略。

> [!note] 你应该知道
> Spinning Up 的现有实现无法让只完成部分训练的智能体从中断处继续训练。如果你认为这个功能很重要，请告诉我们，或者考虑把它作为一个动手改进项目！


## 算法输出

每种算法都会保存一次训练运行的超参数配置、学习进度、训练好的智能体和价值函数；如果可能，还会保存环境的副本（方便同时加载智能体和环境）。输出目录包含以下内容：

|                  |                                                                                                                                                                                                                                |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **输出目录结构** |                                                                                                                                                                                                                                |
| `pyt_save/`      | **仅限 PyTorch 实现。** 此目录包含恢复训练好的智能体和价值函数所需的全部内容。 （参见 [下文的 PyTorch 保存详情。](#下文的 PyTorch 保存详情。)）                                                                                |
| `tf1_save/`      | **仅限 TensorFlow 实现。** 此目录包含恢复训练好的智能体和价值函数所需的全部内容。 （参见 [下文的 TensorFlow 保存详情。](#下文的 TensorFlow 保存详情。)）                                                                       |
| `config.json`    | 一个字典，尽可能完整地描述启动训练函数时使用的位置参数和关键字参数。 如果传入了无法序列化为 JSON 的内容，日志记录器应该能够妥善处理，并在配置文件中将其表示为字符串。 注意：该文件仅用于记录。目前不支持通过配置文件启动实验。 |
| `progress.txt`   | 制表符分隔值（TSV）文件，记录训练期间由日志记录器保存的指标， 例如 `Epoch`、`AverageEpRet` 等。                                                                                                                                |
| `vars.pkl`       | pickle 文件，用于存放需要保存的算法状态相关信息。 目前所有算法都只用它保存环境的副本。                                                                                                                                         |

> [!note] 你应该知道
> 有时由于环境对象无法被 pickle 序列化，保存环境会失败，`vars.pkl` 因而是空的。已知较旧版本 Gym 中的 Box2D 环境存在这个问题，它们无法通过这种方式保存。


> [!note] 你应该知道
> 自 2020 年 1 月 30 日起，保存目录结构略有变化。此前，TensorFlow 计算图保存在 `simple_save/` 文件夹中；现在已改为 `tf1_save/`。


> [!note] 你应该知道
> 这里唯一可能需要你“手动”使用的文件是 `config.json`。智能体测试工具会从 `tf1_save/` 或 `pyt_save/` 目录加载所需内容，绘图工具则会解释 `progress.txt` 的内容；这些才是与相应输出交互的正确工具。`config.json` 没有专门的工具支持，它只用于记录，以便你忘记某次实验使用了哪些超参数时能够重新确认。


### PyTorch 保存目录说明

<div id="下文的 PyTorch 保存详情。">

`pyt_save` 目录包含：

</div>

|                       |                                                                                                                                                       |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Pyt_Save 目录结构** |                                                                                                                                                       |
| `model.pt`            | 使用 `torch.save` 创建的文件，本质上是经过 pickle 序列化的 PyTorch `nn.Module`。 加载它可以将训练好的智能体恢复为带有 `act` 方法的 ActorCritic 对象。 |

### TensorFlow 保存目录说明

<div id="下文的 TensorFlow 保存详情。">

`tf1_save` 目录包含：

</div>

|                       |                                                                                                                                                                     |
|-----------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **TF1_Save 目录结构** |                                                                                                                                                                     |
| `variables/`          | 包含 TensorFlow Saver 输出的目录。参见 [TensorFlow SavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md) 文档。 |
| `model_info.pkl`      | 包含相关信息的字典（从键到张量名称的映射），用于帮助我们在加载后解析已保存的模型。                                                                                  |
| `saved_model.pb`      | 一个 protocol buffer 文件，是 [TensorFlow SavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md) 所必需的。      |

## 保存目录的位置

默认情况下，实验结果会保存在与 Spinning Up 软件包相同的目录中，位于名为 `data` 的文件夹下：

<div class="parsed-literal">

spinningup/  
**data/**  
...

docs/  
...

spinup/  
...

LICENSE setup.py

</div>

你可以修改 `spinup/user_config.py` 中的 `DEFAULT_DATA_DIR`，更改默认结果目录。

## 加载并运行训练好的策略

### 环境成功保存时

如果环境与智能体一起成功保存，就可以通过以下命令方便地观察训练好的智能体在环境中采取行动：

<div class="parsed-literal">

python -m spinup.run test_policy path/to/output_directory

</div>

可以使用以下几个选项：

<div class="option">

-l L, --len=L, default=0

*整数*。测试回合／轨迹／rollout 的最大长度。默认值 0 表示不设最大回合长度，回合仅在智能体到达环境中的终止状态时结束。（注意：将 L 设为 0 并不能阻止由 TimeLimit 包装器包装的 Gym 环境在达到预设最大回合长度时结束。）

</div>

<div class="option">

-n N, --episodes=N, default=100

*整数*。让智能体运行的测试回合数量。

</div>

<div class="option">

-nr, --norender

不将测试回合渲染到屏幕上。这时，`test_policy` 只会打印各回合的回报和长度。（适用情形：渲染器拖慢了测试，而你只想快速了解智能体的表现，并不特别需要观看它的行动。）

</div>

<div class="option">

-i I, --itr=I, default=-1

*整数*。这个选项适用于一种特殊情况：软件包原样提供的算法不支持该情况，但稍作修改即可实现。比如，你可能希望观察训练过程中多个不同阶段的智能体（例如第 50、100、150 次迭代时）。日志记录器可以在这些不同阶段保存智能体快照，供之后运行和观察。在这种情况下，可以使用该参数指定要运行哪个迭代阶段的快照。但再次强调：Spinning Up 的算法默认只保存最近一次智能体快照，并覆盖旧快照。

该参数的默认值表示“使用最新的快照”。

若要修改算法，使其保存多个快照，请在算法中找到以下代码（所有算法中都有）：

``` python
logger.save_state({'env': env}, None)
```

将其修改为：

``` python
logger.save_state({'env': env}, epoch)
```

然后务必将 `save_freq` 设为合理的值（例如，如果保持默认值 1，每个快照都会在输出目录中产生一个 `save` 文件夹，很快就会堆积大量文件夹）。

</div>

<div class="option">

-d, --deterministic

这是另一个特殊选项，仅用于 SAC。Spinning Up 中的 SAC 实现训练的是随机策略，但评估时使用动作分布的确定性 *均值*。`test_policy` 默认使用 SAC 训练得到的随机策略；如果要观察确定性的均值策略（即 SAC 正确的评估策略），应设置此确定性标志。其他算法不会使用此标志。

</div>

### 找不到环境时的错误

如果环境没有成功保存，`test_policy.py` 很可能会崩溃，并显示类似以下内容：

<div class="parsed-literal">

Traceback (most recent call last):  
File "spinup/utils/test_policy.py", line 153, in \<module\>  
run_policy(env, get_action, args.len, args.episodes, not(args.norender))

File "spinup/utils/test_policy.py", line 114, in run_policy  
"and we can't run the agent in it. :( nn Check out the readthedocs " +

AssertionError: Environment not found!

> It looks like the environment wasn't saved, and we can't run the agent in it. :(
>
> Check out the readthedocs page on Experiment Outputs for how to handle this situation.

</div>

在这种情况下，观察智能体的表现会稍微麻烦一些，但只要能够轻松重新创建环境，就仍然可以做到。请在 IPython 中尝试：

``` text
>>> from spinup.utils.test_policy import load_policy_and_env, run_policy
>>> import your_env
>>> _, get_action = load_policy_and_env('/path/to/output_directory')
>>> env = your_env.make()
>>> run_policy(env, get_action)
Logging data to /tmp/experiments/1536150702/progress.txt
Episode 0    EpRet -163.830      EpLen 93
Episode 1    EpRet -346.164      EpLen 99
...
```

### 使用训练好的价值函数

`test_policy.py` 工具无法帮助你查看训练好的价值函数。如果想使用它们，就需要自行进行一些检查。对于 PyTorch，应使用 `torch.load` 加载保存的模型文件，再查看各算法的文档，了解 ActorCritic 对象包含哪些模块。对于 TensorFlow，应使用 [restore_tf_graph](../utils/logger_zh.md) 函数加载保存的计算图，再查看各算法的文档，了解保存了哪些函数。

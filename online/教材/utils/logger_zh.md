---
tags: [SpinningUp, 深度强化学习, 中文教材]
source_rst: docs/utils/logger_zh.rst
source_github: https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/docs/utils/logger_zh.rst
---
# 日志工具



## 使用日志记录器

Spinning Up 提供了基本的日志记录工具，分别由 [Logger](../utils/logger_zh.md) 和 [EpochLogger](../utils/logger_zh.md) 两个类实现。`Logger` 类提供了绝大多数基本功能，用于保存诊断指标、超参数配置、训练运行状态及训练好的模型。`EpochLogger` 类在此基础上添加了一层轻量封装，便于统计每个 epoch 内、以及不同 MPI 工作进程之间的诊断指标均值、标准差、最小值和最大值。

> [!note] 你应该知道
> Spinning Up 中的所有算法实现都使用 `EpochLogger`。


### 示例

首先，看一个简单示例，了解 `EpochLogger` 如何追踪一个诊断指标：

``` text
>>> from spinup.utils.logx import EpochLogger
>>> epoch_logger = EpochLogger()
>>> for i in range(10):
        epoch_logger.store(Test=i)
>>> epoch_logger.log_tabular('Test', with_min_and_max=True)
>>> epoch_logger.dump_tabular()
-------------------------------------
|     AverageTest |             4.5 |
|         StdTest |            2.87 |
|         MaxTest |               9 |
|         MinTest |               0 |
-------------------------------------
```

`store` 方法将 `Test` 的全部取值保存到 `epoch_logger` 的内部状态中。随后，调用 `log_tabular` 时，它会根据内部状态中保存的所有 `Test` 取值，计算均值、标准差、最小值和最大值。调用 `log_tabular` 后，内部状态会被清空，以免这些值混入下一个 epoch 的统计结果。最后，调用 `dump_tabular` 将诊断指标写入文件和标准输出。

接下来，我们查看一个完整的训练过程示例，其中嵌入了日志记录器，以展示如何保存配置、保存模型以及记录诊断指标：

``` python
import numpy as np
import tensorflow as tf
import time
from spinup.utils.logx import EpochLogger


def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)


# Simple script for training an MLP on MNIST.
def train_mnist(steps_per_epoch=100, epochs=5, 
                lr=1e-3, layers=2, hidden_size=64, 
                logger_kwargs=dict(), save_freq=1):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    # Load and preprocess MNIST data
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28*28) / 255.0

    # Define inputs & main outputs from computation graph
    x_ph = tf.placeholder(tf.float32, shape=(None, 28*28))
    y_ph = tf.placeholder(tf.int32, shape=(None,))
    logits = mlp(x_ph, hidden_sizes=[hidden_size]*layers + [10], activation=tf.nn.relu)
    predict = tf.argmax(logits, axis=1, output_type=tf.int32)

    # Define loss function, accuracy, and training op
    y = tf.one_hot(y_ph, 10)
    loss = tf.losses.softmax_cross_entropy(y, logits)
    acc = tf.reduce_mean(tf.cast(tf.equal(y_ph, predict), tf.float32))
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Prepare session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Setup model saving
    logger.setup_tf_saver(sess, inputs={'x': x_ph}, 
                                outputs={'logits': logits, 'predict': predict})

    start_time = time.time()

    # Run main training loop
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            idxs = np.random.randint(0, len(x_train), 32)
            feed_dict = {x_ph: x_train[idxs],
                         y_ph: y_train[idxs]}
            outs = sess.run([loss, acc, train_op], feed_dict=feed_dict)
            logger.store(Loss=outs[0], Acc=outs[1])

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state(state_dict=dict(), itr=None)

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('Acc', with_min_and_max=True)
        logger.log_tabular('Loss', average_only=True)
        logger.log_tabular('TotalGradientSteps', (epoch+1)*steps_per_epoch)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    train_mnist()
```

在这个示例中，请注意：

- 第 19 行使用 [logger.save_config](../utils/logger_zh.md) 将超参数配置保存为 JSON 文件。
- 第 42、43 行使用 [logger.setup_tf_saver](../utils/logger_zh.md)，配置日志记录器以保存计算图中的关键元素。
- 第 54 行通过 [logger.store](../utils/logger_zh.md) 将诊断指标保存到日志记录器的内部状态中。
- 第 58 行通过 [logger.save_state](../utils/logger_zh.md) 每个 epoch 保存一次计算图。
- 第 61–66 行使用 [logger.log_tabular](../utils/logger_zh.md) 和 [logger.dump_tabular](../utils/logger_zh.md) 将每个 epoch 的诊断指标写入文件。请注意，传入 [logger.log_tabular](../utils/logger_zh.md) 的键与传入 [logger.store](../utils/logger_zh.md) 的键相同。

### 日志记录与 PyTorch

前面的示例使用的是 TensorFlow。对于 PyTorch，除了第 42–43 行之外，其余部分都相同：你需要以 `logger.setup_pytorch_saver` 代替 `logger.setup_tf_saver`，并向它传入 [一个 PyTorch 模块](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) （即正在训练的神经网络）作为参数。

`logger.save_state` 的行为与 TensorFlow 版本相同：每次调用都会保存该 PyTorch 模块的最新版本。

### 日志记录与 MPI

> [!note] 你应该知道
> 在强化学习中，若使用 MPI 对梯度和／或其他关键量取平均，一些算法便很容易并行化。Spinning Up 的日志记录器专门设计为能正确适应 MPI：只有 rank 为 0 的进程会向标准输出和文件写入内容。但如果使用 `EpochLogger`，其他进程的信息并不会丢失：无论数据在哪个进程中通过 `store` 存入 `EpochLogger`，它们都会参与诊断指标的均值、标准差、最小值和最大值的计算。


## 日志记录器类

### API：spinup.utils.logx.Logger

源代码：https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/spinup/utils/logx.py

#### Logger

A general-purpose logger.

Makes it easy to save diagnostics, hyperparameter configurations, the state of a training run, and the trained model.

#### Logger.__init__(self, output_dir=None, output_fname='progress.txt', exp_name=None)

Initialize a Logger.

Args:  
output_dir (string): A directory for saving results to. If  
`None`, defaults to a temp directory of the form `/tmp/experiments/somerandomnumber`.

output_fname (string): Name for the tab-separated-value file  
containing metrics logged throughout a training run. Defaults to `progress.txt`.

exp_name (string): Experiment name. If you run multiple training  
runs and give them all the same `exp_name`, the plotter will know to group them. (Use case: if you run the same hyperparameter configuration with multiple random seeds, you should give them all the same `exp_name`.)

#### Logger.log(self, msg, color='green')

Print a colorized message to stdout.

#### Logger.log_tabular(self, key, val)

Log a value of some diagnostic.

Call this only once for each diagnostic quantity, each iteration. After using `log_tabular` to store values for each diagnostic, make sure to call `dump_tabular` to write them out to file and stdout (otherwise they will not get saved anywhere).

#### Logger.save_config(self, config)

Log an experiment configuration.

Call this once at the top of your experiment, passing in all important config vars as a dict. This will serialize the config to JSON, while handling anything which can't be serialized in a graceful way (writing as informative a string as possible).

Example use:

``` python
logger = EpochLogger(**logger_kwargs)
logger.save_config(locals())
```

#### Logger.save_state(self, state_dict, itr=None)

Saves the state of an experiment.

To be clear: this is about saving *state*, not logging diagnostics. All diagnostic logging is separate from this function. This function will save whatever is in `state_dict`---usually just a copy of the environment---and the most recent parameters for the model you previously set up saving for with `setup_tf_saver`.

Call with any frequency you prefer. If you only want to maintain a single state and overwrite it at each call with the most recent version, leave `itr=None`. If you want to keep all of the states you save, provide unique (increasing) values for 'itr'.

Args:  
state_dict (dict): Dictionary containing essential elements to  
describe the current state of training.

itr: An int, or None. Current iteration of training.

#### Logger.setup_tf_saver(self, sess, inputs, outputs)

Set up easy model saving for tensorflow.

Call once, after defining your computation graph but before training.

Args:  
sess: The Tensorflow session in which you train your computation  
graph.

inputs (dict): A dictionary that maps from keys of your choice  
to the tensorflow placeholders that serve as inputs to the computation graph. Make sure that *all* of the placeholders needed for your outputs are included!

outputs (dict): A dictionary that maps from keys of your choice  
to the outputs from your computation graph.

#### Logger.setup_pytorch_saver(self, what_to_save)

Set up easy model saving for a single PyTorch model.

Because PyTorch saving and loading is especially painless, this is very minimal; we just need references to whatever we would like to pickle. This is integrated into the logger because the logger knows where the user would like to save information about this training run.

Args:  
what_to_save: Any PyTorch model or serializable object containing  
PyTorch models.

#### Logger.dump_tabular(self)

Write all of the diagnostics from the current iteration.

Writes both to stdout, and to the output file.


> <div class="automethod">
>
> spinup.utils.logx.Logger.\_\_init\_\_
>
> </div>

### API：spinup.utils.logx.EpochLogger

源代码：https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/spinup/utils/logx.py

#### EpochLogger

A variant of Logger tailored for tracking average values over epochs.

Typical use case: there is some quantity which is calculated many times throughout an epoch, and at the end of the epoch, you would like to report the average / std / min / max value of that quantity.

With an EpochLogger, each time the quantity is calculated, you would use

``` python
epoch_logger.store(NameOfQuantity=quantity_value)
```

to load it into the EpochLogger's state. Then at the end of the epoch, you would use

``` python
epoch_logger.log_tabular(NameOfQuantity, **options)
```

to record the desired values.

#### EpochLogger.__init__(self, *args, **kwargs)

#### EpochLogger.store(self, **kwargs)

Save something into the epoch_logger's current state.

Provide an arbitrary number of keyword arguments with numerical values.

#### EpochLogger.log_tabular(self, key, val=None, with_min_and_max=False, average_only=False)

Log a value or possibly the mean/std/min/max values of a diagnostic.

Args:  
key (string): The name of the diagnostic. If you are logging a  
diagnostic whose state has previously been saved with `store`, the key here has to match the key you used there.

val: A value for the diagnostic. If you have previously saved  
values for this key via `store`, do *not* provide a `val` here.

with_min_and_max (bool): If true, log min and max values of the  
diagnostic over the epoch.

average_only (bool): If true, do not log the standard deviation  
of the diagnostic over the epoch.

#### EpochLogger.get_stats(self, key)

Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.


## 加载已保存的模型（仅限 PyTorch）

要加载由 Spinning Up 的 PyTorch 实现保存的 actor-critic 模型，执行：

``` python
ac = torch.load('path/to/model.pt')
```

使用这种方式加载 actor-critic 模型后，至少可以预期它具有一个 `act` 方法，用于根据观测值从策略中采样动作：

``` python
actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))
```

## 加载已保存的计算图（仅限 TensorFlow）

### API：spinup.utils.logx.restore_tf_graph

源代码：https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/spinup/utils/logx.py

#### restore_tf_graph(sess, fpath)

Loads graphs saved by Logger.

Will output a dictionary whose keys and values are from the 'inputs' and 'outputs' dict you specified with logger.setup_tf_saver().

Args:  
sess: A Tensorflow session. fpath: Filepath to save directory.

Returns:  
A dictionary mapping from keys to tensors in the computation graph loaded from `fpath`.


如果用这种方法恢复由 Spinning Up 的 TensorFlow 实现保存的计算图，至少可以预期其中包含以下内容：

<table>
<thead>
<tr class="header">
<th>键</th>
<th>值</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td><code>x</code></td>
<td>用于输入状态的 TensorFlow 占位符。</td>
</tr>
<tr class="even">
<td><p><code>pi</code></p></td>
<td><p>在 <code>x</code> 中所给状态的条件下，<br />
从智能体的策略中采样一个动作。</p></td>
</tr>
</tbody>
</table>

通常还会保存与该算法相关的价值函数。要了解具体算法还会保存哪些内容，请参阅对应的算法文档页面。

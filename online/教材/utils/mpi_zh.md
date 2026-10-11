---
tags: [SpinningUp, 深度强化学习, 中文教材]
source_rst: docs/utils/mpi_zh.rst
source_github: https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/docs/utils/mpi_zh.rst
---
# MPI 工具



## MPI 核心工具

### API：spinup.utils.mpi_tools

源代码：https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/spinup/utils/mpi_tools.py

#### mpi_fork(n, bind_to_core=False)

Re-launches the current script with workers linked by MPI.

Also, terminates the original process that launched it.

Taken almost without modification from the Baselines function of the [same name](https://github.com/openai/baselines/blob/master/baselines/common/mpi_fork.py).

Args:  
n (int): Number of process to split into.

bind_to_core (bool): Bind each MPI process to a core.

#### msg(m, string='')

#### proc_id()

Get rank of calling process.

#### allreduce(*args, **kwargs)

#### num_procs()

Count active MPI processes.

#### broadcast(x, root=0)

#### mpi_op(x, op)

#### mpi_sum(x)

#### mpi_avg(x)

Average a scalar or vector over MPI processes.

#### mpi_statistics_scalar(x, with_min_and_max=False)

Get mean/std and optional min/max of scalar x across MPI processes.

Args:  
x: An array containing samples of the scalar to produce statistics  
for.

with_min_and_max (bool): If true, return min and max of x in  
addition to mean and std.


## MPI 与 PyTorch 的配套工具

`spinup.utils.mpi_pytorch` 提供了一些工具，便于在多个 MPI 进程之间执行数据并行的 PyTorch 优化。其两个主要环节是同步模型参数，以及在自适应优化器使用梯度之前对梯度求平均。此外，它还包含一种临时性的变通修复办法，用来解决每个进程中的 PyTorch 实例都试图占用过多线程、进而相互争抢资源的问题。

这些工具的典型使用流程如下：

1)  在训练脚本开始时，调用 `setup_pytorch_for_mpi()`。（用于避免线程相互争抢的问题。）
2)  构建好一个 PyTorch 模块后，调用 `sync_params(module)`。
3)  随后在梯度下降过程中，在反向传播之后调用 `mpi_avg_grads`，如下所示：

``` python
optimizer.zero_grad()
loss = compute_loss(module)
loss.backward()
mpi_avg_grads(module)   # averages gradient buffers across MPI processes!
optimizer.step()
```

### API：spinup.utils.mpi_pytorch

源代码：https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/spinup/utils/mpi_pytorch.py

#### setup_pytorch_for_mpi()

Avoid slowdowns caused by each separate process's PyTorch using more than its fair share of CPU resources.

#### mpi_avg_grads(module)

Average contents of gradient buffers across MPI processes.

#### sync_params(module)

Sync all parameters of module across all MPI processes.


## MPI 与 TensorFlow 的配套工具

`spinup.utils.mpi_tf` 包含一些工具，便于在多个 MPI 进程之间使用 AdamOptimizer。这种实现方式多少有些临时拼凑的意味；如果你需要更完善、适用范围更广的方案，可以考虑 [horovod](https://github.com/uber/horovod)。

### API：spinup.utils.mpi_tf

源代码：https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/spinup/utils/mpi_tf.py

#### flat_concat(xs)

#### assign_params_from_flat(x, params)

#### sync_params(params)

#### sync_all_params()

Sync all tf variables across MPI processes.

#### MpiAdamOptimizer

Adam optimizer that averages gradients across MPI processes.

The compute_gradients method is taken from Baselines [MpiAdamOptimizer](https://github.com/openai/baselines/blob/master/baselines/common/mpi_adam_optimizer.py). For documentation on method arguments, see the Tensorflow docs page for the base [AdamOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer).

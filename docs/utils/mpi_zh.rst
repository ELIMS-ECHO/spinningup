====================
MPI 工具
====================

.. contents:: 目录

MPI 核心工具
====================

.. automodule:: spinup.utils.mpi_tools
    :members:


MPI 与 PyTorch 的配套工具
============================

``spinup.utils.mpi_pytorch`` 提供了一些工具，便于在多个 MPI 进程之间执行数据并行的 PyTorch 优化。其两个主要环节是同步模型参数，以及在自适应优化器使用梯度之前对梯度求平均。此外，它还包含一种临时性的变通修复办法，用来解决每个进程中的 PyTorch 实例都试图占用过多线程、进而相互争抢资源的问题。

这些工具的典型使用流程如下：

1) 在训练脚本开始时，调用 ``setup_pytorch_for_mpi()``。（用于避免线程相互争抢的问题。）

2) 构建好一个 PyTorch 模块后，调用 ``sync_params(module)``。

3) 随后在梯度下降过程中，在反向传播之后调用 ``mpi_avg_grads``，如下所示：

.. code-block:: python

    optimizer.zero_grad()
    loss = compute_loss(module)
    loss.backward()
    mpi_avg_grads(module)   # averages gradient buffers across MPI processes!
    optimizer.step()


.. automodule:: spinup.utils.mpi_pytorch
    :members:

MPI 与 TensorFlow 的配套工具
==============================

``spinup.utils.mpi_tf`` 包含一些工具，便于在多个 MPI 进程之间使用 AdamOptimizer。这种实现方式多少有些临时拼凑的意味；如果你需要更完善、适用范围更广的方案，可以考虑 `horovod`_。

.. _`horovod`: https://github.com/uber/horovod

.. automodule:: spinup.utils.mpi_tf
    :members:

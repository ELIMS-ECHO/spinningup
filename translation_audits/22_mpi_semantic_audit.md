# 第22篇《MPI Tools》逐句语义、源码 API 与 Sphinx 构建审计

## 源文件与版本

- 上游英文原文：`openai/spinningup/docs/utils/mpi.rst`，blob `054c4de692f57bf706eceb853759b35565bd7ecd`，48 行。
- 中文译文：`tianyueni78-cyber/spinningup` → `learning-edition-zh` → `docs/utils/mpi_zh.rst`，blob `74229af4c3e47de34bf7a08047bd1edcc97d84ff`，48 行。
- 首次完整译文提交：`827ff80f96f6464adf8852c765895f4c3a3d102e`。
- 独立 Sphinx 构建及 API 真源校验提交：`40af7f51c58a2569994b15093020f305780b75c6`。
- 核对办法：直接读取上游英文源文件与仓库中文文件，按每句独立语义、条件和执行顺序核验；命令与 API 路径用文本程序逐字符对照。实际构建另由 GitHub Actions 执行，避免用“结构一致”冒充“语义准确”。

## 逐句及独立技术命题核对（39 项）

| 编号 | 英文行 | 中文行 | 核验的原文含义／限制 | 结果 |
|---:|:---:|:---:|---|---|
| 01 | 1–3 | 1–3 | 英文标题 MPI Tools 译作 MPI 工具；同时保留 RST 标题装饰线结构。 | 一致 |
| 02 | 5 | 5 | 保留 .. contents:: 文档内目录，仅翻译目录的显示标签。 | 一致 |
| 03 | 7 | 7 | Core MPI Utilities 单独分为核心 MPI 工具小节，没有漏掉该部分。 | 一致 |
| 04 | 9–10 | 9–10 | 核心工具文档由 automodule:: spinup.utils.mpi_tools 指向真实 Python 模块。 | 一致 |
| 05 | 9–10 | 9–10 | :members: 指令保留，要求自动列出模块成员而非只显示标题。 | 一致 |
| 06 | 13 | 13 | MPI + PyTorch Utilities 翻译为 MPI 与 PyTorch 的配套工具。 | 一致 |
| 07 | 15 | 15 | 模块名 spinup.utils.mpi_pytorch 保持逐字符一致。 | 一致 |
| 08 | 15 | 15 | 该模块提供在多个 MPI 进程上运行数据并行 PyTorch 优化的辅助工具。 | 一致 |
| 09 | 15 | 15 | 两个主要环节之一是跨进程同步参数。 | 一致 |
| 10 | 15 | 15 | 另一主要环节是在自适应优化器使用梯度之前对梯度进行跨进程平均。 | 一致 |
| 11 | 15 | 15 | 模块另提供一种较为临时的变通方法，而非彻底解决方案。 | 一致 |
| 12 | 15 | 15 | 需要规避的问题是每个 MPI 进程的 PyTorch 实例都可能创建过多 CPU 线程。 | 一致 |
| 13 | 15 | 15 | 过多线程使不同进程彼此争抢资源，造成性能降低。 | 一致 |
| 14 | 17 | 17 | 下列三步仅描述这些 MPI + PyTorch 工具的一种典型使用模式。 | 一致 |
| 15 | 19 | 19 | 第 1 步必须放在训练脚本开始处。 | 一致 |
| 16 | 19 | 19 | 调用的准确函数为 setup_pytorch_for_mpi()。 | 一致 |
| 17 | 19 | 19 | 这一调用的目的在于缓解不同进程内线程数过多导致的相互争抢问题。 | 一致 |
| 18 | 21 | 21 | 第 2 步是在构建一个 PyTorch module 之后。 | 一致 |
| 19 | 21 | 21 | 对刚构建的 module 调用 sync_params(module) 同步参数。 | 一致 |
| 20 | 23 | 23 | 第 3 步发生在梯度下降时。 | 一致 |
| 21 | 23 | 23 | 应在 loss.backward() 反向传播完成之后调用 mpi_avg_grads。 | 一致 |
| 22 | 23 | 23 | 调用 mpi_avg_grads 的目的是在优化器更新之前，对 MPI 进程的梯度做平均。 | 一致 |
| 23 | 25–31 | 25–31 | 保留 code-block:: python 指令。 | 一致 |
| 24 | 25–31 | 25–31 | 按原文保持 optimizer.zero_grad() 的清梯度动作。 | 一致 |
| 25 | 25–31 | 25–31 | 按原文保持 loss = compute_loss(module) 的损失计算动作。 | 一致 |
| 26 | 25–31 | 25–31 | 按原文保持 loss.backward() 在 mpi_avg_grads(module) 之前。 | 一致 |
| 27 | 25–31 | 25–31 | 按原文保持 mpi_avg_grads(module) 在 optimizer.step() 之前。 | 一致 |
| 28 | 25–31 | 25–31 | 代码内英文注释说明对 MPI 进程间梯度缓冲区取平均，作为代码原文保留未擅改。 | 一致 |
| 29 | 25–31 | 25–31 | 整段示例代码与英文源文件逐字符一致。 | 一致 |
| 30 | 34–35 | 34–35 | PyTorch 的 API 自动文档指向 spinup.utils.mpi_pytorch 模块而不是外部伪实现。 | 一致 |
| 31 | 34–35 | 34–35 | 该模块的 :members: 选项保留。 | 一致 |
| 32 | 37 | 37 | MPI + Tensorflow Utilities 章节译为 MPI 与 TensorFlow 的配套工具。 | 一致 |
| 33 | 39 | 39 | 模块名 spinup.utils.mpi_tf 保留。 | 一致 |
| 34 | 39 | 39 | 该模块提供在多个 MPI 进程中使用 AdamOptimizer 的辅助工具。 | 一致 |
| 35 | 39 | 39 | 原文称实现方案有些 hacky，译文保留其临时拼凑、并非通用方案的限制。 | 一致 |
| 36 | 39 | 39 | 如果需要更成熟、更通用的方案，原文建议考虑 horovod，不是强制要求。 | 一致 |
| 37 | 39 | 39 | 没有把 TensorFlow 版本宣称为与 Horovod 性能相同。 | 一致 |
| 38 | 41 | 41 | Horovod 外部链接仍为 https://github.com/uber/horovod。 | 一致 |
| 39 | 43–44 | 43–44 | TensorFlow 的 API 自动文档指向 spinup.utils.mpi_tf 模块，保留 :members:。 | 一致 |

## 可复核的程序化验证

- 原文与中文的文件均为 48 行；主体章节和三个 `.. automodule::` 指令完全对应。
- 所有 3 个模块路径逐字符一致：`spinup.utils.mpi_tools`、`spinup.utils.mpi_pytorch`、`spinup.utils.mpi_tf`。
- 参数同步、梯度下降的三步列表仍是 1) → 2) → 3)，没有交换执行顺序。
- 从 `.. code-block:: python` 至 `spinup.utils.mpi_pytorch` 指令前的完整代码示例字节级文本相同，保留示例的 5 条可执行语句和原文内联注释。
- 原文与译文唯一外部 URL 均为 `https://github.com/uber/horovod`。
- 三个 API 来源文件已在当前 GitHub 分支直接读取成功：
  - `spinup/utils/mpi_tools.py`，blob `0d032129f6def08ac4e41f63f736d59a497c8dba`；
  - `spinup/utils/mpi_pytorch.py`，blob `1b46e4efb54bea77ed82117fee97f24b1ff29114`；
  - `spinup/utils/mpi_tf.py`，blob `96cbcf5e06eec16727b3dc10806a376e3670046c`。
- 翻译的是 HTML 页面源码里的叙述文本；`automodule` 后的实际 API 自动文档依旧来自未修改的原始 Python docstrings。

## GitHub Actions 真实构建与生成内容验证

- 严格构建：https://github.com/tianyueni78-cyber/spinningup/actions/runs/38055145991
- 对应版本：commit `40af7f51c58a2569994b15093020f305780b75c6`，结论 `success`。日志包含 `build succeeded`，无 Sphinx `WARNING` 或 `ERROR`。
- 执行的命令：`sphinx-build -b html -n -W --keep-going`，本轮中文独立构建包含第13–22篇的有关页面。
- 工作流使用源仓库中真实的 `mpi_tools.py`、`mpi_pytorch.py`、`mpi_tf.py` 文件，与此前的日志模块一起拷贝到仅用于文档的临时 Python 包，并通过 `sphinx.ext.autodoc` 渲染。
- 工作流在生成 HTML 后以三项真实 docstring 断言验证 API 内容没有缺失：
  - `Re-launches the current script with workers linked by MPI.` 来自 `mpi_tools.py`；
  - `Average contents of gradient buffers across MPI processes.` 来自 `mpi_pytorch.py`；
  - `Sync all tf variables across MPI processes.` 来自 `mpi_tf.py`。
- 为避免导入旧版 `spinup/__init__.py` 后激活整个 TF1/旧版 PyTorch 算法栈，工作流采用独立的包入口并 mock 外部依赖 `tensorflow`、`torch`、`numpy`、`joblib`、`mpi4py`。这种方式验证真实 Python **API 文档生成**，不代表在真实多进程环境中运行优化器或验证分布式训练数值正确性。
- 整站 Sphinx 原配置集成构建仍不在本次通过范围内；工作流沿用前面篇章 `ref.footnote` 未被引用脚注的定向警告豁免，此篇自身没有脚注豁免。

## 最终结论

本篇 39 项独立语义命题通过核对，三组真实模块 API 指令、程序代码、外链与原文一致。GitHub Actions 独立 Sphinx HTML 构建和三组真实源代码 API 文档渲染检查已通过；MPI 运行时多进程功能和原站点全依赖集成构建未做验收。

---
tags: [SpinningUp, 深度强化学习, 中文教材]
source_rst: docs/user/plotting_zh.rst
source_github: https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/docs/user/plotting_zh.rst
---
# 绘制实验结果

Spinning Up 附带一个简单的绘图工具，用于解读实验结果。使用以下命令运行：

<div class="parsed-literal">

python -m spinup.run plot \[path/to/output_directory ...\] \[--legend \[LEGEND ...\]\]  
\[--xaxis XAXIS\] \[--value \[VALUE ...\]\] \[--count\] \[--smooth S\] \[--select \[SEL ...\]\] \[--exclude \[EXC ...\]\]

</div>

**位置参数：**

<div class="option">

logdir

*字符串*。可以指定任意数量的日志目录，也可以指定日志目录路径的前缀（绘图工具会在内部自动补全），从中绘制结果。绘图工具会递归搜索日志目录，查找实验输出。

> [!note] 你应该知道
> 内部自动补全功能非常方便！假设你已经运行了几组实验，目的是比较不同算法的性能，得到以下日志目录结构：
>
> <div class="parsed-literal">
>
> data/  
> bench_algo1/  
> bench_algo1-seed0/ bench_algo1-seed10/
>
> bench_algo2/  
> bench_algo2-seed0/ bench_algo2-seed10/


你可以轻松使用以下命令生成比较 algo1 和 algo2 的图：

<div class="parsed-literal">

python spinup/utils/plot.py data/bench_algo

</div>

绘图工具会依靠自动补全，同时找到 `data/bench_algo1` 和 `data/bench_algo2`。

</div>

</div>

**可选参数：**

<div class="option">

-l, --legend=\[LEGEND ...\]

*字符串*。可选参数，用于指定图例。默认情况下，绘图工具会自动使用 `config.json` 文件中的 `exp_name` 作为图例名称；也可以通过此参数覆盖。只有在为每个将要绘制的目录都提供一个名称时，该参数才有效。（注意：这不一定等于你传入的 logdir 参数个数！请记住，绘图工具会自动补全 logdir 参数：一个日志目录前缀可能匹配多个目录，而你需要为每一个匹配结果提供一条图例文字，除非你已通过下文的筛选或排除规则，从候选目录中去掉了其中一些。）

</div>

<div class="option">

-x, --xaxis=XAXIS, default='TotalEnvInteracts'

*字符串*。选择数据中哪一列用作横轴。

</div>

<div class="option">

-y, --value=\[VALUE ...\], default='Performance'

*字符串*。选择数据中的哪些列作为纵轴绘图。传入多个值会生成多张图。默认值为 `Performance`，但这并不是任何算法实际输出的一列。相反，`Performance` 指代以下两者之一：对于同策略（on-policy）算法，正确的性能指标是 `AverageEpRet`；对于离策略（off-policy）算法，正确的性能指标是 `AverageTestEpRet`。绘图工具会为每个独立日志目录自动判断应使用 `AverageEpRet` 还是 `AverageTestEpRet`。

</div>

<div class="option">

--count

可选标志。默认情况下，绘图工具会对具有相同 `exp_name` 的所有结果的纵轴数值求平均，并显示平均曲线。这些结果通常对应仅随机种子不同、其他设置完全相同的一组实验。如果想分别查看各次运行的曲线，请使用 `--count` 标志。

</div>

<div class="option">

-s, --smooth=S, default=1

*整数*。通过在固定窗口内取平均来平滑数据。此参数指定平均窗口的宽度。

</div>

<div class="option">

--select=\[SEL ...\]

*字符串*。可选的筛选规则：绘图工具仅显示日志目录路径中包含这里所有子字符串的曲线。

</div>

<div class="option">

--exclude=\[EXC ...\]

*字符串*。可选的排除规则：绘图工具仅显示日志目录路径中不包含任何一个给定子字符串的曲线。

</div>

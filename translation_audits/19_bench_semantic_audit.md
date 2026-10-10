# 第19篇 Benchmarks for Spinning Up Implementations：逐句语义与图表构建审计

## 版本与范围

- 英文原文：`openai/spinningup/docs/spinningup/bench.rst`，blob `be0152b6b604cdffb57d0a6f8beb416d8b2f87ce`。
- 中文译文：`tianyueni78-cyber/spinningup` 分支 `learning-edition-zh`，文件 `docs/spinningup/bench_zh.rst`，核验时 blob `55e12f665580941a220f116f4c67efa8fbe54e87`。
- 译文创建提交：`3b7d33b47ba07ed983177153f626321a4aae10ca`。
- Sphinx 构建配置升级提交：`3ebf0ceed95b08523d0f77a493f504f2cc2380ea`，包含图像资产复制。
- 本章属于历史基准测试介绍。评价结论以所翻译的原始资料为准，不推断成 2026 年算法最新性能排名。

## 逐句语义条件核对

按英文每个句子的独立事实或限定拆分，并对原始 10 幅图表的标题、源文件、对齐格式及图注分别审查。行号均为 GitHub 原始文件中的一基编号，不是通用编号。

| # | 原文行 | 中文行 | 对照条件及结论 | 结果 |
|---:|:---:|:---:|---|---|
| 001 | 2 | 2 | 标题保持 Spinning Up 算法实现的基准测试这一限定，而非泛称所有强化学习算法的基准。 | 一致 |
| 002 | 5 | 5 | 目录指令保留（contents），仅将展示名称翻译为“目录”。 | 一致 |
| 003 | 7 | 7 | 基准测试对象是 Spinning Up 的算法实现；评估任务属于 Gym 的 MuJoCo 环境套件。 | 一致 |
| 004 | 7 | 7 | 五个环境的名字和数量严格一致：HalfCheetah、Hopper、Walker2d、Swimmer、Ant。 | 一致 |
| 005 | 9 | 9 | MuJoCo 引用目标 URL 保持原样。 | 一致 |
| 006 | 11 | 11 | 章节讨论的是各环境中的算法性能，不涉及额外实验或方法。 | 一致 |
| 007 | 14 | 14 | HalfCheetah 分类标题及 PyTorch 框架名称正确对应原文。 | 一致 |
| 008 | 17–18 | 17–18 | HalfCheetah/PyTorch 的 SVG 原始相对文件路径及 :align: center 指令均完整保留。 | 一致 |
| 009 | 20 | 20 | HalfCheetah-v3 的版本号、PyTorch 框架及 3M timestep（300 万时间步）实验规模均一致。 | 一致 |
| 010 | 23 | 23 | HalfCheetah 分类标题及 TensorFlow 框架名称正确对应原文。 | 一致 |
| 011 | 26–27 | 26–27 | HalfCheetah/TensorFlow 的 SVG 原始相对文件路径及 :align: center 指令均完整保留。 | 一致 |
| 012 | 29 | 29 | HalfCheetah-v3 的版本号、TensorFlow 框架及 3M timestep（300 万时间步）实验规模均一致。 | 一致 |
| 013 | 33 | 33 | Hopper 分类标题及 PyTorch 框架名称正确对应原文。 | 一致 |
| 014 | 36–37 | 36–37 | Hopper/PyTorch 的 SVG 原始相对文件路径及 :align: center 指令均完整保留。 | 一致 |
| 015 | 39 | 39 | Hopper-v3 的版本号、PyTorch 框架及 3M timestep（300 万时间步）实验规模均一致。 | 一致 |
| 016 | 42 | 42 | Hopper 分类标题及 TensorFlow 框架名称正确对应原文。 | 一致 |
| 017 | 45–46 | 45–46 | Hopper/TensorFlow 的 SVG 原始相对文件路径及 :align: center 指令均完整保留。 | 一致 |
| 018 | 48 | 48 | Hopper-v3 的版本号、TensorFlow 框架及 3M timestep（300 万时间步）实验规模均一致。 | 一致 |
| 019 | 53 | 53 | Walker2d 分类标题及 PyTorch 框架名称正确对应原文。 | 一致 |
| 020 | 56–57 | 56–57 | Walker2d/PyTorch 的 SVG 原始相对文件路径及 :align: center 指令均完整保留。 | 一致 |
| 021 | 59 | 59 | Walker2d-v3 的版本号、PyTorch 框架及 3M timestep（300 万时间步）实验规模均一致。 | 一致 |
| 022 | 62 | 62 | Walker2d 分类标题及 TensorFlow 框架名称正确对应原文。 | 一致 |
| 023 | 65–66 | 65–66 | Walker2d/TensorFlow 的 SVG 原始相对文件路径及 :align: center 指令均完整保留。 | 一致 |
| 024 | 68 | 68 | Walker2d-v3 的版本号、TensorFlow 框架及 3M timestep（300 万时间步）实验规模均一致。 | 一致 |
| 025 | 72 | 72 | Swimmer 分类标题及 PyTorch 框架名称正确对应原文。 | 一致 |
| 026 | 75–76 | 75–76 | Swimmer/PyTorch 的 SVG 原始相对文件路径及 :align: center 指令均完整保留。 | 一致 |
| 027 | 78 | 78 | Swimmer-v3 的版本号、PyTorch 框架及 3M timestep（300 万时间步）实验规模均一致。 | 一致 |
| 028 | 81 | 81 | Swimmer 分类标题及 TensorFlow 框架名称正确对应原文。 | 一致 |
| 029 | 84–85 | 84–85 | Swimmer/TensorFlow 的 SVG 原始相对文件路径及 :align: center 指令均完整保留。 | 一致 |
| 030 | 87 | 87 | Swimmer-v3 的版本号、TensorFlow 框架及 3M timestep（300 万时间步）实验规模均一致。 | 一致 |
| 031 | 91 | 91 | Ant 分类标题及 PyTorch 框架名称正确对应原文。 | 一致 |
| 032 | 94–95 | 94–95 | Ant/PyTorch 的 SVG 原始相对文件路径及 :align: center 指令均完整保留。 | 一致 |
| 033 | 97 | 97 | Ant-v3 的版本号、PyTorch 框架及 3M timestep（300 万时间步）实验规模均一致。 | 一致 |
| 034 | 100 | 100 | Ant 分类标题及 TensorFlow 框架名称正确对应原文。 | 一致 |
| 035 | 103–104 | 103–104 | Ant/TensorFlow 的 SVG 原始相对文件路径及 :align: center 指令均完整保留。 | 一致 |
| 036 | 106 | 106 | Ant-v3 的版本号、TensorFlow 框架及 3M timestep（300 万时间步）实验规模均一致。 | 一致 |
| 037 | 109 | 109 | 实验细节章节独立保留。 | 一致 |
| 038 | 112 | 112 | 每项实验运行 10 个不同的随机种子，未误作总计 10 个种子。 | 一致 |
| 039 | 112 | 112 | 曲线展示在训练过程中的跨随机种子的性能平均值。 | 一致 |
| 040 | 112 | 112 | 均值使用实线表示。 | 一致 |
| 041 | 112 | 112 | 标准差使用阴影区域表示，没有误译为置信区间或标准误。 | 一致 |
| 042 | 114 | 114 | 同策略算法的性能指标是每个 epoch 收集的轨迹批次的平均轨迹回报。 | 一致 |
| 043 | 114 | 114 | 离策略算法的性能每 10,000 个交互步测量一次，未误译成每 10,000 个 epoch。 | 一致 |
| 044 | 114 | 114 | 离策略算法测试使用确定性策略。 | 一致 |
| 045 | 114 | 114 | 对于 SAC，明确替换为动作分布的均值策略。 | 一致 |
| 046 | 114 | 114 | 离策略测试不添加动作噪声，原文限制得到保留。 | 一致 |
| 047 | 114 | 114 | 每次离策略评估运行恰好 10 条测试轨迹。 | 一致 |
| 048 | 114 | 114 | 报告的是上述测试轨迹的平均回报。 | 一致 |
| 049 | 116 | 116 | 同策略算法在策略网络和价值函数网络上都使用 (64, 32) 的隐藏层大小。 | 一致 |
| 050 | 116 | 116 | 同策略两类网络均使用 tanh 激活单元。 | 一致 |
| 051 | 116 | 116 | 离策略算法使用 (256, 256) 的网络大小。 | 一致 |
| 052 | 116 | 116 | 离策略算法使用 ReLU（原文 relu）激活单元。 | 一致 |
| 053 | 118 | 118 | 同策略算法在每次批量更新前收集 4,000 步 agent-environment 交互。 | 一致 |
| 054 | 118 | 118 | 离策略算法每次梯度下降步骤使用大小为 100 的 minibatch。 | 一致 |
| 055 | 120 | 120 | 除明确提及的设置以外，其余超参数采用 Spinning Up 实现的默认值。 | 一致 |
| 056 | 120 | 120 | 引导读者在各个算法文档页面查询默认超参数细节。 | 一致 |
| 057 | 122 | 122 | 学习曲线使用 11 个 epoch 宽度的窗口取平均进行平滑。 | 一致 |
| 058 | 124 | 124 | 警示框（admonition）存在，标题“你应该知道”对应原文。 | 一致 |
| 059 | 126 | 126 | DDPG、TD3、SAC 的 Spinning Up 实现在原作者当时与文献最佳报告结果“大致持平”，未夸大为严格超越。 | 一致 |
| 060 | 126 | 126 | 因此原作者判断这三种算法的 Spinning Up 实现可用于研究。 | 一致 |
| 061 | 128 | 128 | VPG、TRPO、PPO 实现总体弱于当时各自报告的最好表现，保留“略弱”的程度词。 | 一致 |
| 062 | 128 | 128 | 该弱势部分由未采用某些常见工程技巧导致。 | 一致 |
| 063 | 128 | 128 | 缺失技巧的两个例子：观测值归一化与价值回归目标归一化。 | 一致 |
| 064 | 128 | 128 | 针对研究中的比较，作者建议使用 OpenAI Baselines 的 TRPO 或 PPO 实现。 | 一致 |
| 065 | 130 | 130 | OpenAI Baselines 的 GitHub URL 严格保留，没有替换为一般主页。 | 一致 |
| 066 | 133 | 133 | PyTorch 与 TensorFlow 对比章节标题完整对应。 | 一致 |
| 067 | 137 | 136 | 介绍的是各个算法的两种框架实现之间的正面对比图，不是仅比较框架速度。 | 一致 |
| 068 | 139 | 138 | VPG 直接对比的链接标签对应同一算法。 | 一致 |
| 069 | 149 | 148 | VPG 对比页面的原始 ../spinningup/bench_vpg.html 目标保持不变。 | 一致 |
| 070 | 141 | 140 | PPO 直接对比的链接标签对应同一算法。 | 一致 |
| 071 | 150 | 149 | PPO 对比页面的原始 ../spinningup/bench_ppo.html 目标保持不变。 | 一致 |
| 072 | 143 | 142 | DDPG 直接对比的链接标签对应同一算法。 | 一致 |
| 073 | 151 | 150 | DDPG 对比页面的原始 ../spinningup/bench_ddpg.html 目标保持不变。 | 一致 |
| 074 | 145 | 144 | TD3 直接对比的链接标签对应同一算法。 | 一致 |
| 075 | 152 | 151 | TD3 对比页面的原始 ../spinningup/bench_td3.html 目标保持不变。 | 一致 |
| 076 | 147 | 146 | SAC 直接对比的链接标签对应同一算法。 | 一致 |
| 077 | 153 | 152 | SAC 对比页面的原始 ../spinningup/bench_sac.html 目标保持不变。 | 一致 |

## 原文图像与目标文件完整性

- 上游 `.. figure::` 项共 **10** 个，译文 **10** 个，路径顺序和文件名完全一致。
- 10 个 SVG 文件在 `learning-edition-zh` 分支实际存在（10/10 已逐个从 GitHub fetch 成功）。
- 两版 10 个图注分别保留 `-v3` 环境版本、`PyTorch/TensorFlow` 实现区分、`3M timestep`（300 万时间步）。
- 原文 `.. _...` 引用目标共 **7** 项：MuJoCo、OpenAI Baselines 和 5 个算法对比页。译文这 7 个 URL/路径按顺序全部一致。
- 五个算法对比的目标 `bench_vpg.rst`、`bench_ppo.rst`、`bench_ddpg.rst`、`bench_td3.rst` 和 `bench_sac.rst` 已在目标仓库分别核对存在。这确认仓库源文件存在，不等于网站公开 URL 已实际部署或线上可访问。
- 14 个标题层级节点（包含第一级、第二级和 10 个环境×框架的子标题）顺序对应，`:align: center` 保留。
- 所有 10 幅图的 SVG 仍引用原始图片资产，没有替换成示意图或重绘数据；因此图表数据未在翻译中改动。

## Sphinx 实际 HTML 编译

- 工作流：`.github/workflows/zh-rst-check.yml`，构建命令 `sphinx-build -b html -n -W --keep-going`。
- 原图在 `docs/images/plots/pyt` 和 `docs/images/plots/tf1`；由于隔离检查将中文 RST 移至 `/tmp/spinningup-zh-rst`，工作流同时把 SVG 拷贝至 `/tmp/images/plots`，以保持相对引用的原意。
- **验收运行：** https://github.com/tianyueni78-cyber/spinningup/actions/runs/38053905806 ，head commit `3ebf0ceed95b08523d0f77a493f504f2cc2380ea`。
- 真实运行状态 **success**，日志明确列出读取与写出 `bench_zh` 页面，以及从 10% 至 100% 复制全部 10 幅 SVG 图片，最后为 `build succeeded`，没有 `WARNING:` 或 `ERROR:`。
- 本流程对前一篇原文自带的未引用脚注保留 `ref.footnote` 类别警告豁免；本章没有新增豁免。此验收证明独立中文 RST 页面、链接引用语法及 SVG 资产能正确完成 Sphinx HTML 构建。**不代表旧版 Spinning Up 原配置下整站集成构建通过，不代表重新运行算法生成测试图，也不代表外链的网络可用性。**

## 最终结论

第 19 篇中文翻译已完成，并通过 77 项逐句独立语义核对；10 幅原始图表与全部 7 个引用目标已通过完整性检查；GitHub Actions 实际 Sphinx 构建通过。归档的结论仅涵盖译文语义、文档资源以及独立文档编译。

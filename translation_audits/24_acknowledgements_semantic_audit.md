# 第24篇《Acknowledgements》翻译逐项审计与 Sphinx 验收

## 原文、译文及已提交版本

- 英文原文：`openai/spinningup/docs/etc/acknowledgements.rst`，blob `17a0639f564cbfe42bf8b68fe56be9dfc5e075c4`，9 行。
- 中文译文：`tianyueni78-cyber/spinningup` 分支 `learning-edition-zh`，`docs/etc/acknowledgements_zh.rst`，blob `b41e62bbaeeac96cfba7ed802ea20ad2a0d4ac92`，9 行。
- 中文版首次提交：`c46b04b790cc93f88df20ce71989a47e5d2550fb`。
- 实际 Sphinx 构建工作流修改提交：`9bf9323042f65bfda25025734dc52fe523654c2e`。
- 编号依据：英文站点 `docs/index.rst` 的 toctree 在 Utilities Docs 的 `utils/run_utils` 后依次为 `etc/acknowledgements`、`etc/author`；本篇对应第24篇，不混入第25篇。

## 逐句／独立语义核对（32 项）

按源文的主句、从句、否定和限定分别核对，并将致谢名单内每个英文姓名单列。注意这份审计是翻译对照，不是对致谢对象现职或外部贡献的事实调查。

| 序号 | 英文行号 | 中文行号 | 原文含义与核对重点 | 结果 |
|---:|---:|---:|---|---|
| 01 | 2 | 2 | 英文标题 Acknowledgements 完整对应中文“致谢” | 一致 |
| 02 | 1 | 3 | RST 标题上下装饰线均存在，适配中文显示宽度 | 一致 |
| 03 | 5 | 5 | 致谢对象是帮助该项目启动的众多贡献者，包含项目起步的背景限定 | 一致 |
| 04 | 5 | 5 | 感谢参与该软件 beta 测试的人，没有将 beta 测试改为正式发布 | 一致 |
| 05 | 5 | 5 | 感谢对教学材料提出反馈的人，反馈对象没有省略 | 一致 |
| 06 | 5 | 5 | 感谢为支持这次发布而改进 Spinning Up 代码依赖的人，保留依赖改进与发布的因果限定 | 一致 |
| 07 | 5 | 5 | 还感谢以其他方式支持项目的人，保留开放性兜底条件 | 一致 |
| 08 | 5 | 5 | 由于参与者在不同阶段加入，名单可能不详尽，不得宣称完全覆盖 | 一致 |
| 09 | 5 | 5 | 认为自己应被致谢的人可以主动联系项目方，保留邀请语气 | 一致 |
| 10 | 7 | 7 | 致谢名单不分先后，不能暗示姓名排序属于贡献排名 | 一致 |
| 11 | 7 | 7 | 第 1 位姓名 Alex Ray 英文拼写与原文一致、位置顺序一致 | 一致 |
| 12 | 7 | 7 | 第 2 位姓名 Amanda Askell 英文拼写与原文一致、位置顺序一致 | 一致 |
| 13 | 7 | 7 | 第 3 位姓名 Ben Garfinkel 英文拼写与原文一致、位置顺序一致 | 一致 |
| 14 | 7 | 7 | 第 4 位姓名 Christy Dennison 英文拼写与原文一致、位置顺序一致 | 一致 |
| 15 | 7 | 7 | 第 5 位姓名 Coline Devin 英文拼写与原文一致、位置顺序一致 | 一致 |
| 16 | 7 | 7 | 第 6 位姓名 Daniel Zeigler 英文拼写与原文一致、位置顺序一致 | 一致 |
| 17 | 7 | 7 | 第 7 位姓名 Dylan Hadfield-Menell 英文拼写与原文一致、位置顺序一致 | 一致 |
| 18 | 7 | 7 | 第 8 位姓名 Ge Yang 英文拼写与原文一致、位置顺序一致 | 一致 |
| 19 | 7 | 7 | 第 9 位姓名 Greg Khan 英文拼写与原文一致、位置顺序一致 | 一致 |
| 20 | 7 | 7 | 第 10 位姓名 Jack Clark 英文拼写与原文一致、位置顺序一致 | 一致 |
| 21 | 7 | 7 | 第 11 位姓名 Jonas Rothfuss 英文拼写与原文一致、位置顺序一致 | 一致 |
| 22 | 7 | 7 | 第 12 位姓名 Larissa Schiavo 英文拼写与原文一致、位置顺序一致 | 一致 |
| 23 | 7 | 7 | 第 13 位姓名 Leandro Castelao 英文拼写与原文一致、位置顺序一致 | 一致 |
| 24 | 7 | 7 | 第 14 位姓名 Lilian Weng 英文拼写与原文一致、位置顺序一致 | 一致 |
| 25 | 7 | 7 | 第 15 位姓名 Maddie Hall 英文拼写与原文一致、位置顺序一致 | 一致 |
| 26 | 7 | 7 | 第 16 位姓名 Matthias Plappert 英文拼写与原文一致、位置顺序一致 | 一致 |
| 27 | 7 | 7 | 第 17 位姓名 Miles Brundage 英文拼写与原文一致、位置顺序一致 | 一致 |
| 28 | 7 | 7 | 第 18 位姓名 Peter Zokhov 英文拼写与原文一致、位置顺序一致 | 一致 |
| 29 | 7 | 7 | 第 19 位姓名 Pieter Abbeel 英文拼写与原文一致、位置顺序一致 | 一致 |
| 30 | 9 | 9 | 感谢 Pieter Abbeel 位于 Berkeley 的研究团队，不误称为大学整体的所有研究人员 | 一致 |
| 31 | 9 | 9 | 感谢 Center for Human-Compatible AI，机构英文名称和含义未丢失 | 一致 |
| 32 | 9 | 9 | 上述团队与中心所提供的是对介绍 Spinning Up 的演示报告（presentations）的反馈，不扩大为算法验证或论文审稿 | 一致 |

## 保真和结构完整性检查

- 英文原文及中文译文均为 9 行，正文三个非空自然语言段落均有对应，原文的括号式联系邀请完整保留。
- 英文名单共 19 人。逐个对照并验证姓与名字符串完全匹配、没有缺席或调序：Alex Ray；Amanda Askell；Ben Garfinkel；Christy Dennison；Coline Devin；Daniel Zeigler；Dylan Hadfield-Menell；Ge Yang；Greg Khan；Jack Clark；Jonas Rothfuss；Larissa Schiavo；Leandro Castelao；Lilian Weng；Maddie Hall；Matthias Plappert；Miles Brundage；Peter Zokhov；Pieter Abbeel。
- 最后一段再次出现 Pieter Abbeel，语义是感谢其伯克利研究团队；与名单中的 Pieter Abbeel 不属于重复误译。
- 机构 `Center for Human-Compatible AI` 保留完整英文名称，仅在括号中附中文释义，未改变机构身份。
- 原文无 URL、代码、API、图像、数学公式或外部文件引用，因此不存在本章需要保持的这些元素；没有凭空增加链接和技术章节。

## GitHub Actions：Sphinx HTML 真实构建

- 工作流 `.github/workflows/zh-rst-check.yml` 新增对 `docs/etc/acknowledgements_zh.rst` 的 push 触发、临时复制及 Sphinx toctree 条目。
- 实际构建记录：https://github.com/tianyueni78-cyber/spinningup/actions/runs/38104056907
- 对应代码提交 `9bf9323042f65bfda25025734dc52fe523654c2e`，实际状态 `completed`，结果 `success`。
- Sphinx 执行 `sphinx-build -b html -n -W --keep-going`，日志明确显示 `reading sources... acknowledgements_zh`、`writing output... acknowledgements_zh` 以及 `build succeeded`，无 `WARNING:` 或 `ERROR:`。
- 该构建继续覆盖此前译好的中文页面（第13–23篇）以及新增第24篇；由于采用独立临时配置，不代表原网站完整配置已经集成构建通过。
- 与此前历史章节相同，独立工作流仍针对原文未被正文引用的编号脚注有 `ref.footnote` 定向警告豁免；本篇原文没有脚注，因此并非靠豁免隐藏此篇解析问题。

## 结论

第24篇译文全文完整，32 项独立语义／姓名对照全部通过。19 人姓名顺序保持不变，机构与致谢贡献范围忠实于原文，已提交 GitHub，Sphinx 独立 HTML 构建通过。尚未执行原站点完整集成构建。

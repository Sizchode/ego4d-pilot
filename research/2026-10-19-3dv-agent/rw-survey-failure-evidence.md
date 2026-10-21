# [RW survey](https://docs.google.com/document/d/1X13Fi5gbiWErC4c5hkfoe4iqzudjUiAnuxgsgVudbok/edit?tab=t.0)：已有方法能修到哪一步

固定照片的三维重建｜证据与待验证问题

我们研究的是：给定同一组照片，能否用不额外训练的 agent，把现有模型估计的相机位置和三维表面修得更好。先看 benchmark 和已有失败证据，再决定需要开发什么工具。

先看一个真实例子：[VGGT-Ω](https://github.com/facebookresearch/vggt-omega) 论文图6展示，[DA3](https://github.com/ByteDance-Seed/Depth-Anything-3) 在重复雪地纹理的视频中几乎没有预测出相机运动；另一个航拍例子重建出重复塔楼。Ω 改善了这些结果 [[S5]](https://arxiv.org/html/2605.15195v1#S4.SS2)。这说明错误存在，也说明换更强的几何模型就能解决一部分；两个例子并未标为我们五套 benchmark 的场景。

## 1. Benchmark 已经告诉我们什么

采用 [DA3](https://github.com/ByteDance-Seed/Depth-Anything-3) 发布的五套评测子集：DTU 看物体表面，ETH3D 看真实室内外，ScanNet++ 看复杂室内，7Scenes 看手持室内序列，HiRoom 看合成室内。统一输入和评测代码，避免把不同论文的分数直接拼成排行榜 [[S1]](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/BENCHMARK.md)。

| 数据 / 证据 | 已有结果说明什么 |
| --- | --- |
| DTU：π³ 表6 [[S2]](https://arxiv.org/html/2507.13347v3#S4.SS4) | 更换参考帧会影响 [VGGT](https://github.com/facebookresearch/vggt) 的结果，π³ 已明显减轻这种敏感性。 |
| ETH3D：[GLUEMAP](https://arxiv.org/html/2605.26103v1#S4) 表1 [[S3]](https://arxiv.org/html/2605.26103v1#S4) | π³ 的相机 AUC@3°：36.1；加 BA 后55.1；[GLUEMAP](https://arxiv.org/html/2605.26103v1#S4) 为76.9。证明相机优化有效，未直接证明表面更准。 |
| ETH3D / ScanNet++：[GGPT](https://github.com/ChenYutongTHU/GGPT) [[S4]](https://arxiv.org/html/2603.11174v1)[[S4b]](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Chen_GGPT_Geometry-Grounded_Point_CVPR_2026_supplemental.pdf) | 有修复前后误差图；ETH3D 八视图中，[VGGT](https://github.com/facebookresearch/vggt) 点图 AUC@5cm 从23到47，是完整修复流程的收益。 |
| 7Scenes / HiRoom：[DA3](https://github.com/ByteDance-Seed/Depth-Anything-3) 发布结果 [[S1]](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/BENCHMARK.md) | 给定真值相机时分数更高，但不能据此断言错误来自模糊、材质或匹配。 |

特别注意：给 [DA3](https://github.com/ByteDance-Seed/Depth-Anything-3) 真值相机，还会改变深度预测和尺度处理。因此两种模式的分数差，不能当成“只修相机就能获得的提升” [[S9]](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/src/depth_anything_3/bench/evaluator.py#L182)。现有材料也没有给出覆盖这五套数据、逐场景标注错误原因的完整清单。

## 2. 重建与修复 baseline

直接重建：[VGGT（CVPR 2025）](https://github.com/facebookresearch/vggt)、[VGGT-Ω（CVPR 2026）](https://github.com/facebookresearch/vggt-omega)、[π³-X](https://github.com/yyfz/Pi3)（2025年12月工程更新）、[MapAnything（3DV 2026）](https://map-anything.github.io/)、[DA3（ICLR 2026）](https://github.com/ByteDance-Seed/Depth-Anything-3)，从照片预测相机、深度或三维点。它们决定我们应该超过什么水平。

几何修复：MP-SfM、[GLUEMAP](https://arxiv.org/html/2605.26103v1#S4) 改善相机估计；[GGPT](https://github.com/ChenYutongTHU/GGPT) 用照片得到的稀疏几何约束修正稠密点云。它们说明“重建后再修一次”已有有效方法，必须直接比较 [[S3]](https://arxiv.org/html/2605.26103v1#S4)[[S4]](https://arxiv.org/html/2603.11174v1)[[S10]](https://arxiv.org/html/2504.20040v1)。

## 3. 待实测核对的失败线索

Agent 相关：[AREA3D](https://github.com/TianlingXu/AREA3D) 通过获取新视角改善重建；[GRASE](https://arxiv.org/abs/2610.21498) 修复可编辑的仿真场景。我们使用固定照片，评价相机和真实场景几何。但“检查—调用工具—修复”本身不能作为新的技术贡献。

| 值得检查的错误 | 已有方法与待验证问题 |
| --- | --- |
| 相机位置不准，多个视图拼不齐 | BA / [GLUEMAP](https://arxiv.org/html/2605.26103v1#S4) 已能优化。先实测哪些错位仍未解决，再决定是否需要新工具及 VLM 选择。 |
| 相机大致对齐，但表面重影或错形 | [GGPT](https://github.com/ChenYutongTHU/GGPT) 已有修复证据。先分析其剩余失败，再决定工具需补什么；VLM 定位疑点的收益待测。 |
| 相似物体被当成同一个 | Ω 与 DG++ 有案例 [[S5]](https://arxiv.org/html/2605.15195v1#S4.SS2)[[S6]](https://arxiv.org/html/2412.05826v2)，尚非五套数据的普查。仅在开发集确认后接入图像对验证；VLM 的实例判断须超过已有分类器和几何验证。 |

暂不优先做：π³ 已明显减轻参考帧依赖，[π³-X](https://github.com/yyfz/Pi3) 已改进网格伪影与置信度 [[S7]](https://github.com/yyfz/Pi3#introducing-pi3x-engineering-update)。透明物体是 π³ 报告的局限，但尚无对应五套测试集的发生率与成功修复证据；识别玻璃后删掉点，并不等于恢复了玻璃的几何。

## 4. 从失败证据形成方案

先复现 baseline，在开发场景统计反复出现的失败及已有方法的修复效果；据此选核心问题、开发并验证 toolkit，最后设计 agent workflow。上表是文献线索，不是已确定的工具清单。

工具有效后，再比较固定流程、几何选择器、一次性 VLM、多步 VLM 及无图像版本，检验 VLM 是否更会选择修复目标和操作。工具收益、看图收益和执行反馈收益分开报告。

模型权重冻结，允许场景内几何优化。具体工具与 workflow 尚待失败分析确定；VLM 的增量仍是假设。下一步是 baseline 复现与失败档案，见“实验计划”标签页。

# Training-Free 3D 重建诊断与修复

Related-work survey｜2026-10-19｜研究设想，尚无实验结论

## 1. 项目定位与一个例子

我们研究一个无需额外训练的重建修复系统：预训练视觉语言模型（VLM）查看原图和几何诊断，选择现有工具，依据执行反馈继续修复。Training-free 指不为本任务训练或微调模型，允许几何优化和在开发集上调流程。Reasoning 指诊断错误、选择操作和检查结果；最终要证明的是重建质量提高。

例如，两扇相似的窗被错误匹配，导致两面墙拼到一起。系统先检查可疑匹配，再找同一批输入中同时拍到两扇窗的照片；若周边标识和空间关系支持它们属于不同窗户，就调用工具重验对应、解除错误关联并重新求解。若证据仍不足，就继续检查或停止。这个示意例子说明工具如何参与推理，不代表已经实现。

## 2. Related work 按解决的问题分类

| 类别 | 代表工作 | 核心做法与本题关系 |
| --- | --- | --- |
| 几何纠错与恢复 | [Heinly 2014](https://www.cs.unc.edu/~jheinly/publications/eccv2014-heinly.pdf)；[Progressive SfM](https://arxiv.org/html/1803.07349v1) | 检查冲突并纠错、拆分或重算重建。几何方法本身已有反馈与恢复能力。 |
| 学习匹配与几何 | [Doppelgangers++](https://arxiv.org/html/2412.05826v2)；[MVDG](https://arxiv.org/html/2610.01098v1)；[GLUEMAP](https://arxiv.org/html/2605.26103v2) | 前两者用图像判断共视关系；后者融合局部预测与全局求解。它们是可复用工具，也是必须比较的强方法。 |
| 图像修复 Agent | [AgenticIR](https://openreview.net/pdf?id=3RLxccFPHz)（ICLR 2025） | 诊断图像退化、安排工具、检查后重新规划；其质量分析 VLM 有微调。流程与本题接近，目标是 2D 图像。 |
| 3D 工具推理 | [Think3D](https://arxiv.org/html/2601.13029v5)（2026 预印本） | 调用重建和视角工具回答空间问题，含 training-free 流程与 RL 版本。其主要输出是答案，我们评价相机和几何。 |
| 重建与仿真 Agent | [LiteReality-Agent](https://arxiv.org/html/2610.01863v1)；[Agentic Real2Sim](https://arxiv.org/html/2607.19190v5)（2026 预印本） | 已有场景编辑验证、分割与跟踪重试；目标为可交互场景或仿真片段，不等同于通用多视图相机修复。 |

这些工作说明，“会调工具、会检查结果”已有充分先例。我们要验证的是这套做法能否有效修复多视图相机与几何，而不是声称首次提出 Agent 闭环。

## 3. 我们的系统与要验证的增量

候选贡献是一个针对常见重建错误的 training-free 诊断与修复流程。输入为图像、初始重建和运行记录；工具返回可疑图像及匹配、图连接、重投影误差和深度一致性。VLM 选择检查对象、操作与参数，执行后更新状态，再决定继续、撤销或停止。输出是修复后的相机和几何，操作记录用于分析。

工具负责匹配、分割、优化和融合；Agent 负责按当前证据组织它们。关键假设是：同一套工具在不同错误上需要不同操作顺序，执行反馈能帮助修正最初判断。实验要区分收益来自工具更强，还是选择与反馈更有效；不要求证明每种错误只有 VLM 能解决。

## 4. 从错误类型选择工具与 benchmark

| 错误类型 | 候选工具 | Benchmark | 检验内容 |
| --- | --- | --- | --- |
| 对应不足、相机未注册 | 补充匹配、重选初始化、重估相机 | [PhotoTourism](https://github.com/ubc-vision/image-matching-benchmark) | 位姿误差、正确注册数；参考来自重建，有自身偏差。 |
| 重复结构误匹配 | 共视验证、解除错误关联、局部重建 | [Doppelgangers / VisymScenes](https://arxiv.org/html/2412.05826v2) | 错误关联识别与错误合并；另查相机精度及覆盖损失。 |
| 表面孔洞、错误深度 | 重选源视图、深度估计与融合 | [ETH3D](https://www.eth3d.net/data/schoeps2017cvpr.pdf) / [DTU](https://roboimagedata2.compute.dtu.dk/data/text/multiViewCVPR2014.pdf) | 精度与完整度；固定相机与重估相机分开评价。 |
| 动态前景污染 | 运动分割、屏蔽后重新匹配与求解 | [TUM](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) / [Bonn](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/) 动态序列 | 轨迹误差与覆盖；RGB 和 RGB-D 协议分开。 |

主线可用 PhotoTourism 与重复结构专项检验相机修复，ETH3D 补充表面几何；动态作为可选扩展。表中是候选处理路径，不是各数据集的失败频率统计。图像对标签不能代替匹配点或相机真值；DTU 提供的相机不能作为未知相机恢复结果。

## 5. Baseline 与评测如何支撑论文

完整系统：[COLMAP](https://demuc.de/colmap/)、GLUEMAP。专项方法：重复结构比较 Doppelgangers++、MVDG、[LFOE](https://openaccess.thecvf.com/content/CVPR2025/papers/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.pdf) 接入同一求解后端；若做动态，再加入同传感器条件的 [RoMo](https://openaccess.thecvf.com/content/ICCV2025/html/Goli_RoMo_Robust_Motion_Segmentation_Improves_Structure_from_Motion_ICCV_2025_paper.html) 或 [DynaSLAM](https://arxiv.org/abs/1806.05620) 等对照。AgenticIR 等用于借鉴流程，其原始任务成绩不能直接放进相机榜单。

核心消融：固定同一初始重建和运行记录，在同工具、同图像与同计算预算下，比较固定重试流程、VLM 一次性制定完整计划、VLM 根据反馈多步调整；后两者使用同一模型。再去掉图像输入或执行反馈，检查诊断与反馈各有多少作用。报告相机误差与正确注册数、几何精度与完整度，以及总时间和模型调用成本。

开发与测试按场景分开；工具、提示词和停止规则在测试前固定。测试参考答案仅用于离线评分，不参与动作选择。尚待确定的是候选匹配如何定位、修改接受与撤销规则、误差门槛及计算预算。重投影误差变小不保证真实几何正确，不能单靠它批准修复。

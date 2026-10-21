# Training-Free Agentic 3D Reconstruction

任务 → Benchmark → Baseline → Toolkit｜2026-10-19

## 1. 我们到底要改善什么

给定同一组静态场景 RGB 图像，Agent 调用并调整重建工具，改善最终相机和稠密几何。比如一间房被重建成双层墙：可能是相机错位，也可能是深度不一致或融合错误。先看重建结果和证据，再决定修哪一步；不能预先把所有问题都当成匹配错误。

主任务是未知相机位姿的多视图重建，输入仅 RGB，输出相机、深度及融合点云。Training-free 指冻结 VLM 和几何模型权重；允许几何优化，以及在开发集迭代工具和流程。主要结果必须包含几何精度与完整度，注册率只是辅助。

## 2. 先选能评价这个任务的 benchmark

近期 benchmark 有两层：新评测套件，以及套件沿用的经典数据。DA3（ICLR 2026）公开了五数据集的位姿与几何评测；SpatialBench（2026）统一了 19 个来源数据集及多种任务，但并非每个数据集都有点云真值。主实验采用 DA3 的以下五套数据及发布协议：

| 数据与发布规模 | 评价对象 | 为什么需要它 |
| --- | --- | --- |
| DTU：22 场景×49视图；位姿另用13×64 | 物体表面几何；独立位姿子集 | 检验局部形状和完整度，不能只把相机修好。 |
| ETH3D：11 场景 | 室内外扫描几何与相机 | 检验场景级精度及跨环境表现。 |
| ScanNet++：20 场景 | 真实室内扫描几何与相机 | 检验复杂室内的完整重建。 |
| 7Scenes：7 场景 | RGB-D 融合参考与相机 | 检验手持室内采集条件下的重建。 |
| HiRoom：24 场景 | 合成室内几何与相机 | 补充合成环境，观察真实/合成差异。 |

以上规模以 DA3 当前发布版为准：ETH3D 有图像过滤，ScanNet++ 有位姿重标定，HiRoom 的发布评测数与论文总数不同。固定版本、图像清单和对齐规则，不能与原数据集完整官方榜单混称；ScanNet++ 相机参考来自重建，7Scenes 来自 KinectFusion，均有误差。

新数据 RealX3D（2026）专门覆盖低光、模糊、遮挡与烟雾，值得用于后续错误分析；但其 NTIRE 挑战提供真值相机、主评新视角图像，不能直接当作本任务的未知位姿几何榜单。是否加入主实验取决于完整几何数据与协议的可用性。

## 3. 对照哪些近期方法

| 方法 | 年份 / 版本 | 比较角色 |
| --- | --- | --- |
| VGGT；VGGT-Ω | CVPR 2025；CVPR 2026 | 联合预测相机和几何；覆盖较新的强模型。 |
| π³ / π³-X | π³：ICLR 2026；X：2025.12 工程更新 | 另一类多视图几何模型；实验选 X 公开版本。 |
| MapAnything | 3DV 2026（预印本 2025） | 通用度量重建；本实验只给 RGB。 |
| Depth Anything 3 | ICLR 2026（预印本 2025） | 深度与相机联合重建，也是评测协议来源。 |

上表均有官方代码；年份区分论文与工程更新。Omega 权重需申请，按作者要求使用 416-Reproduction 基准权重。所有方法在同一图像清单、RGB-only 设置和评估器下重跑，报告各自分辨率、耗时和显存；不直接拼接不同论文的分数。

另设 GLUEMAP → DA3 的固定串联对照：前者预测相机，后者用预测相机重建稠密几何。这是我们构造的近期方法组合，非一篇独立论文；用于检验简单串联是否已经足够。

Related work：MP-SfM（2025）与 GLUEMAP（2026）侧重 SfM，AREA3D（CVPR 2026）获取新视角，GRASE（2026 预印本）用反馈修复可编辑仿真场景。我们评价固定 RGB 输入的相机与稠密几何，不能将通用“检查—修复”循环本身宣称为新意。

## 4. 根据错误开发工具，再验证 Agent

先在与测试按场景隔离的开发集运行 baseline，检查哪些失败确实存在及影响多大，再开发 toolkit。候选问题包括相机错位、深度错估、跨视图不一致，以及融合后的重影、孔洞或浮点；错误频率需要实测，不能由数据集名称推断。

工具随问题设计：相机问题可尝试重匹配和优化；深度问题可尝试局部重估、分辨率或视图组合调整；融合问题可尝试跨视图筛选和重新融合。具体组件、参数范围及预算在开发集迭代，工具无效就修改。测试前冻结工具、提示词、策略及公开输入清单；真值、评分和评测掩码仅在评估端使用。

Workflow：初始重建 → 查看原图与几何状态 → 提出可验证的错误判断 → 执行修复 → 重新检查 → 保留、撤销或停止。运行时只能看输入和可计算的一致性证据，不能访问真值或根据测试得分选择输出。

要验证的贡献是诊断与反馈能帮助 training-free Agent 更有效地使用工具、改善几何。控制实验共享初始重建和工具：固定流程、一次性 VLM 计划、多步 Agent；一次性计划可调用多个工具，但不能根据反馈重规划。统一硬件与总 GPU 秒上限，计入 VLM 和失败尝试，另报 token。去掉图像输入检验视觉增量；若只有工具有效，就不能归功于 Agent。

主表同时报告几何 F-score / Overall 与位姿 AUC@3°/30°，沿用各数据集阈值、掩码和尺度对齐；DTU 用毫米，其他用米，分表报告，不混算原始距离。补充改善/退化比例与成本。尚未实测，因此不预判哪类错误最常见、哪个工具最好或 Agent 一定有效。

参考链接：[DA3（ICLR 2026）](https://github.com/ByteDance-Seed/Depth-Anything-3/blob/main/docs/BENCHMARK.md)；[SpatialBench（2026）](https://github.com/Ropedia/SpatialBench)；[RealX3D（2026）](https://i2wm.github.io/3DRR_2026/)；[VGGT；](https://github.com/facebookresearch/vggt)；[VGGT-Ω](https://github.com/facebookresearch/vggt-omega)；[π³ / π³-X](https://github.com/yyfz/Pi3)；[MapAnything](https://map-anything.github.io/)；[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)；[416-Reproduction](https://github.com/facebookresearch/vggt-omega/blob/main/reproduction.md)；[MP-SfM（2025）](https://github.com/cvg/mpsfm)；[GLUEMAP（2026）](https://github.com/colmap/gluemap)；[AREA3D（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/papers/Xu_AREA3D_Active_Reconstruction_Agent_with_Unified_Feed-Forward_3D_Perception_and_CVPR_2026_paper.pdf)；[GRASE（2026 预印本）](https://arxiv.org/abs/2610.21498)

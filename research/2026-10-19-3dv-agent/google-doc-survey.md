# 3DV + Agent：文献综述与候选研究问题

文献截点：2026-10-19｜研究讨论稿

## 1. 核心判断与问题定义

本文讨论真实多视角重建中的诊断与恢复：给定图像、已有相机/几何估计和可访问的执行记录，系统判断错误的可能来源，选择有成本的诊断或修复操作，并以最终相机和几何质量为目标。它不同于在固定三维场景中回答问题，也不同于生成一个视觉合理的场景。

**当前结论：问题值得研究，但“Agent 读取几何诊断并调用标准工具”尚不足以定义论文贡献。** 传统 SfM 已有恢复、拆分合并、迭代扩图；现代方法已有学习筛边、神经先验融合、可靠性门控；3D 与视觉 Agent 已有反馈、回滚、记忆和策略训练。不能把这些组件的组合直接称为研究空白。

**插件形态与贡献强弱是两个维度。** PixSfM 可以作为现有流程的附加模块，其贡献仍是 featuremetric refinement；LFOE 可以作为筛边模块，其贡献仍是学习到的图表示与过滤。真正需要避免的是：去掉工具名和 Agent 标签后，只剩提示词编排和增加计算量。[PixSfM，ICCV 2021](https://arxiv.org/abs/2108.08291)；[LFOE，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.pdf)。

后续选题不预设必须使用 LLM、必须限定 view graph 或必须采用多 Agent。首先明确现有方法无法处理的几何失败，再判断缺少的是信息、表示、优化还是决策。本文的候选问题是待验证假设，不是已经确认的空白，也不是 pilot 计划。

## 2. 研究版图与任务边界

| 研究线 | 已有能力 | 对候选选题的约束 |
|---|---|---|
| 几何恢复与鲁棒估计 | 定位冲突、删除异常约束、拆分合并、撤销早期估计 | “诊断后修复”本身不是新任务结构 |
| 图构建、筛选与学习 | 自适应补边、网络流选图、学习筛边、多视图消歧 | 不能只与默认 COLMAP 或固定阈值比较 |
| 神经先验与几何优化融合 | 深度/法向先验、局部前馈几何、虚拟 tracks、联合 BA | “失败时换模型再优化”已有强方法 |
| 可靠性与序贯计算 | uncertainty 校准、refinement gate、扰动诊断、动态算法配置 | 需要解释历史和额外诊断的具体价值 |
| 3D 与视觉 Agent | 主动采集、QA、真实场景编辑、real-to-sim、工具策略学习 | 必须区分被修改的对象与反馈的来源 |

图像检索提议匹配对；几何验证建立约束；图筛选选择约束；鲁棒估计从污染约束中求解；恢复撤销或重建已有结果；先验融合增加原始匹配之外的信息。这些阶段不能全部合并成“工具选择”，否则难以解释真正的技术增量。

固定动作接口可以使问题清晰、便于比较，不等于用手写规则解决 corner cases。需要区分动作定义与控制策略：标准 matching、BA 等操作可以固定，何时、何地、为何调用它们，则应有可解释的推断或学习机制。

## 3. 文献线一：传统 SfM 恢复与图处理

### 3.1 错误结构恢复已经是明确研究对象

**Correcting for Duplicate Scene Structure in Sparse 3D Reconstruction（ECCV 2014）**直接研究失败重建中的错误检测与纠正。**Recovering Correct Reconstructions from Indistinguishable Geometry（3DV 2014）**利用几何共现定位致错结构，拆成子模型后重新合并。它们针对重复/不可区分几何，并不等于解决所有失败，但已覆盖“先诊断、再恢复”的基本结构。此处依据原始摘要和机构页面；前者全文访问未成功，不据此推断实现细节。[ECCV 论文](https://www.cs.unc.edu/~jheinly/publications/eccv2014-heinly.pdf)；[3DV 机构记录](https://researchwith.stevens.edu/en/publications/recovering-correct-reconstructions-from-indistinguishable-geometr/)。

**Progressive Structure from Motion（ECCV 2018）**是尤其重要的近邻。它随输入和图拓扑变化复查局部结果，进行重聚类、结构转移、拆分、合并和 reset/re-estimate；全局保留的约束使后续证据有机会纠正早期决策。其输入协议是 progressive，而非任意固定图像集上的通用控制器，但“反馈、可逆性、积累证据后重判”已存在。潜在增量必须落在新的状态表示、目标或决策机制上。[全文](https://arxiv.org/html/1803.07349v1)。

### 3.2 图构建与选择并非只有静态 retrieval

**GraphMatch（3DV 2017）**通过 sample-and-propagate 提出新图像对，几何验证后更新图邻域并继续搜索。它已经把当前计算结果用于后续匹配，主要目标是高效发现有效边，而非解释复杂失败。[全文](https://arxiv.org/html/1710.01602v1)。

**View-graph Selection Framework for SfM（ECCV 2018）**使用 minimum-cost network flow，结合图与几何/context priors 选择子图，兼顾歧义、效率和准确率。预算式图选择不能被概括为简单阈值规则。[正式论文](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Rajvi_Shah_View-graph_Selection_Framework_ECCV_2018_paper.pdf)。

**View-graph Construction Framework for Robust and Efficient SfM（Pattern Recognition 2021）**从已验证的生成树出发，利用局部重建、相机注册产生新的相对几何并迭代扩图。因此“执行局部重建获得证据，再更新图”也已有先例。[出版社原文](https://www.sciencedirect.com/science/article/pii/S003132032030515X)。

### 3.3 鲁棒估计与近期恢复近邻

**1DSfM（ECCV 2014）**从一维投影中的顺序冲突识别异常相对平移；**COLMAP（CVPR 2016）**本身包含增量注册、三角化、过滤和 BA；**GLOMAP（ECCV 2024）**通过旋转平均、相机与点联合 global positioning 和 BA 求解。它们都不是等待外部 Agent 才处理错误的“正常流程”。[1DSfM](https://research.cs.cornell.edu/1dsfm/)；[COLMAP](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf)；[GLOMAP](https://arxiv.org/html/2407.20219v1)。

**TC-SfM（2022 预印本）**利用 track-community/context 识别歧义部分，分块重建并进行一致性合并。作者指出弱连接可能导致部分相机被隔离或丢失，提示“删掉不可靠约束”与“保持可恢复性”存在冲突。[全文](https://arxiv.org/html/2206.05866v1)。

**Robust Global SfM via View Graph Pruning（2026-08-22 预印本）**递归划分图并实际运行子图 GLOMAP，再通过一致性和跨块 PnP/RANSAC 验证剪边。它已经是执行驱动的诊断，而非只读取静态统计。其算子顺序和门槛预定义，未研究异质诊断计算的长期选择价值；这个区别仅针对该文，不是全领域空白证明。[全文](https://arxiv.org/html/2608.22054v1)。

## 4. 文献线二：学习修图与神经几何融合

### 4.1 视觉信息与图学习已有直接应用

**Doppelgangers（ICCV 2023）及 Doppelgangers++（CVPR 2025）**区分外观相似、实际属于不同三维表面的图像。后者使用 MASt3R 的 3D-aware features 和 Transformer 分类器，并集成到 SfM。视觉高层线索识别假匹配，并非 VLM 独有能力。[Doppelgangers](https://openaccess.thecvf.com/content/ICCV2023/html/Cai_Doppelgangers_Learning_to_Disambiguate_Images_of_Similar_Structures_ICCV_2023_paper.html)；[Doppelgangers++](https://arxiv.org/html/2412.05826v2)。

**Learning to Filter Outlier Edges in Global SfM / LFOE（CVPR 2025）**把原图的边变为 line graph 节点，以相对几何、预估旋转和 DINOv2 特征学习过滤。它既有跨数据评价，也有 oracle filter。该 oracle 只针对特定删边准则，作者指出处理退化配置仍可能超过它，因此不能把它当作全 SfM 的严格上界。[正式论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.pdf)。

**VGPA（ECCV 2026；arXiv 2026-09-08）**通过 edge-conditioned GNN 从 noisy view graphs 估计全局相机，推理时还有场景一致性 adaptation，并可重新纳入被删相机。“学习图处理 + test-time adaptation + 相机恢复”已经有组合；它并非纯一次前向过程。[全文](https://arxiv.org/html/2609.09491v1)。

**MVDG（2026-10-01 预印本）**使用 VGGT 多视图特征消歧，并通过 confidence-based early exit 分配计算。其 early-exit confidence 与最终深层输出的一致性有关，不自动等于真实几何正确率。多视图上下文和自适应计算也需要与此类方法比较。[全文](https://arxiv.org/html/2610.01098v1)。

### 4.2 强 hybrid reconstruction 会改变恢复空间

**MASt3R-SfM（3DV 2025）**将 learned local geometry/matches 与全局对齐结合，是完整重建路线，而不只是可调用的初始化工具。本处依据原始摘要及作者页面。[论文记录](https://arxiv.org/abs/2409.19152)。

**MP-SfM（CVPR 2025）**融合深度、法向先验及不确定性，使用 lifting、depth consistency、扩展 BA 与先验细化，针对低视差、低重叠等困难。其启发是：部分失败来自约束不足，需要增加并正确融合信息；换一个调度器未必能解决。先验 uncertainty 也需要校准。[全文](https://arxiv.org/html/2504.20040v1)。

**GLUEMAP（CVPR 2026）**将局部 feed-forward stars、motion averaging、真实和虚拟 tracks 的 augmented BA 结合起来。这已经覆盖大量“传统 SfM 失败时借助前馈几何”的设想。其剩余限制包括对局部模型质量的依赖，不能因此推导通用恢复已解决。[全文](https://arxiv.org/html/2605.26103v2)。

相邻进展还有 **VGGSfM（CVPR 2024）**的可微 tracks/camera/BA、**G-MASt3R-SfM（2026）**的图剪枝和多阶段优化，以及 **Glob3R（2026）**的跨窗口 tracks 与全局优化。后两者在本轮主要用于补充方法版图，不据摘要宣称性能优劣。[VGGSfM](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_VGGSfM_Visual_Geometry_Grounded_Deep_Structure_From_Motion_CVPR_2024_paper.pdf)；[G-MASt3R-SfM](https://arxiv.org/html/2606.22856v1)；[Glob3R](https://arxiv.org/abs/2607.09225)。

## 5. 文献线三：可靠性、主动诊断与算法控制

### 5.1 从模型置信度到决策可靠性，仍有距离

**LoRA3D（ICLR 2025）**使用鲁棒点图对齐、置信度重估与筛选伪标签，再做场景 LoRA 适配。重建—验证—修正循环已有具体非语言机制，其价值在几何目标与置信度更新的关系。[全文](https://arxiv.org/html/2412.07746v1)。

**Trust3R（ICML 2026）**在冻结 MASt3R 后增加 evidential head 和 gated residual head，预测点分布并选择性修正。它处理单点 XYZ 的协方差，不显式建模跨像素相关性。局部可靠性不能自动解释为全局相机图失败概率。[全文](https://arxiv.org/html/2605.19539v1)。

**GeoCond（2026-09-16 预印本）**从视差、重叠、基线、旋转和 confidence 等特征预测 pose reliability 与 refinement gate，用于是否 BA、图剪枝、旋转平均和采集停止。uncertainty 可用自监督信号训练，但 BA gate 的训练标签利用 GT pose error 判断优化收益；推理不需要 GT。作者将完整 SfM/SLAM 轨迹验证留作后续。这是“读几何诊断再决定调用优化器”的直接近邻。[全文](https://arxiv.org/html/2609.18465v1)。

**ReVar3R（2026-10-06 预印本）**指出不同坐标规范中的点图方差会混入非误差因素，因此先做鲁棒 Sim(3) 对齐再计算多次预测分歧。核心可无标签，完整融合/校准版本使用标签；作者明确承认无法发现稳定系统偏差。多次预测都同意，仍可能一起错。[全文](https://arxiv.org/html/2610.07883v1)。

**VGGT uncertainty audit（2026-06-15 预印本）**提醒置信度含义取决于输出阶段和实际评价目标，BA 后不能直接沿用优化前的不确定性解释。DUSt3R 官方 confidence loss 的形式也不意味着它就是“相机正确概率”。[VGGT audit](https://arxiv.org/html/2606.16479v1)；[DUSt3R 官方实现](https://github.com/naver/dust3r/blob/main/dust3r/losses.py)。

### 5.2 序贯工具选择有更早的方法基础

**Dynamic Algorithm Configuration（JAIR 2022）**已经研究按执行状态动态改变算法配置。若观测是低维统计、动作只是几个 matcher/BA 配置，首先应考虑 algorithm control，而非默认需要语言模型。[论文](https://arxiv.org/abs/2205.13881)。

**Deep Adaptive Design（ICML 2021）**根据历史 design–observation 学习后续实验选择，以全序列信息增益为目标；**dual control**研究动作既改善系统又提供信息的双重作用。这些思想与“先做诊断，再决定修复”相关，但不能直接套用：SfM 的诊断工具离散、成本不同、观测相关，且常缺乏显式 likelihood。[DAD](https://proceedings.mlr.press/v139/foster21a.html)；[Dual Control and Online Optimal Experimental Design](https://epubs.siam.org/doi/10.1137/16M1069936)。

因此 belief state、POMDP、信息增益或 RL 这些术语本身不是新贡献。必须说明隐藏量、可用证据、坐标规范、动作副作用和几何目标如何使问题不同。软件状态可读，不代表真实错误原因可知；反过来，也不能仅因存在 Agent 就宣布需要部分可观测模型。

## 6. 文献线四：3D Agent 与视觉工具策略

### 6.1 主动视角与空间 QA：改变观察不等于修复估计

**AREA3D（CVPR 2026）**将重建置信度和 VLM 区域指导组成 utility fields，以候选视角排序和覆盖更新选择采集动作。**Active3D（2025 首稿）**结合 uncertainty、规划与局部 refinement。不能说主动重建都假设流程正常，但它们主要研究采集和表示更新，而非错误 correspondence/相机图恢复。[AREA3D](https://arxiv.org/html/2512.05131v2)；[Active3D](https://arxiv.org/html/2511.20050v1)。

**Think3D（首版 2026-01-19；所读 v5）**在重建后通过变换/渲染回答问题，最新版本每个问题最多一次 reconstruction。**SpatialClaw（2026-06）**通过持久 Python 状态组织空间工具推理，并研究接口差别。**Skill-3D（2026-06）**提取 skills/lessons，经 SFT + GRPO 学习工具使用。它们最终评价主要是 QA，并未证明底层 reconstruction 被修复。[Think3D](https://arxiv.org/html/2601.13029v5)；[SpatialClaw](https://arxiv.org/html/2606.13673v1)；[Skill-3D](https://arxiv.org/html/2606.07436v2)。

**Explore until Confident**使用 conformal prediction 研究探索停止；**E3VS-Bench**提供 3DGS 环境的主动搜索任务；**MAG-3D**组织 planner、grounding 和 coding verification。它们说明停止规则、证据获取、程序验证均已有研究，同时必须区分“在重建环境里找答案”和“恢复环境自身”。本段以摘要及作者页面核查为主。[Explore until Confident](https://explore-eqa.github.io/)；[E3VS](https://arxiv.org/abs/2604.17969)；[MAG-3D](https://arxiv.org/abs/2604.09167)。

### 6.2 真实重建与 real-to-sim：最直接的 Agent 近邻

**Agentic Real2Sim（首版 2026-07-21；所读 v1/v5）**使用标定、多视角/深度及机器人轨迹，critic 可重选 mask、关键帧和 tracking 初始化，再根据回放修改。其对象是可回放的交互 episode，不是通用未知相机图。v5 已加入 PSNR、SSIM、ADD-S 和机器人用途，不能说只有 VLM judge；但 replay success 仍有最佳候选和 judge 定义，且 one-prompt 对照缺少相同专用工具，不能据此隔离控制器的贡献。[版本记录](https://arxiv.org/abs/2607.19190)；[v5 全文](https://arxiv.org/html/2607.19190v5)。

**LiteReality-Agent（2026-10-01）**从 RGB-D scan、已知相机、点云与 RoomPlan layout 出发，观察证据、编辑 Room.py，检查碰撞、支撑和 articulation。它主要修复可编辑场景资产；评估使用采集视角，不能当作 held-out novel-view 验证。observe–edit–verify 的真实场景闭环已经存在。[全文](https://arxiv.org/html/2610.01863v1)。

**FACT（首版 2026-09-29；所读 v2）**对 articulated twin 进行定向测量、几何修订、接触/碰撞与动力学拟合。其几何参考包含 image-generated mesh，不能等同独立真实几何 GT。“先测量、再诊断、再修改表示”不能作为独有机制。[全文](https://arxiv.org/html/2609.37067v2)。

**Agentic RSR（首版 2026-10-07；所读 v2）**从视频及前馈重建出发，结合已知机器人 URDF 确定尺度，修改 Blender 场景并做交互检查。其参考相机被冻结；depth feedback 来自先前预测，而非独立传感器真值。它已有重建返工，但不等于恢复相机估计。[全文](https://arxiv.org/html/2610.10479v2)。

**Agentic Building-Aware Satellite GS（2026-09-22）**在有限权重计划中选择配置，依赖候选验证 DSM 指标，只有四个 JAX 场景；不能当作已验证的 GT-free 通用在线恢复。**Harvard SceneAgent**的第一方页面也包含 COLMAP→3DGS 的检查与修正，但本轮未确认独立论文或首次公开日期。[卫星工作](https://arxiv.org/html/2609.25578v1)；[SceneAgent 项目](https://computationalrobotics.seas.harvard.edu/SceneAgent/)。

其他相邻工作包括 **AGILE** 的手物交互重建、**Real2Gym** 的视频到仿真与技能复用、**RoboSimStudio** 的可编辑仿真工具。它们支持拓宽近邻范围，本轮摘要核查不足以判断其所有模块消融。[AGILE](https://agile-hoi.github.io/)；[Real2Gym](https://arxiv.org/abs/2609.37089)；[RoboSimStudio](https://arxiv.org/abs/2610.18835)。

### 6.3 生成编辑与 restoration agents 限制通用框架新颖性

**SAGE、SceneSmith、MUSE**分别强调场景生成及 critics、分层设计和资产物理属性、局部编辑与约束保留。这些任务可以有系统贡献，但不能与从真实图像恢复相机/几何混为一谈。[SAGE](https://arxiv.org/abs/2602.10116)；[SceneSmith](https://arxiv.org/abs/2602.09153)；[MUSE](https://arxiv.org/abs/2606.14168)。

更直接的通用结构近邻是 **RestoreAgent（NeurIPS 2024）**、**AgenticIR（ICLR 2025）**、**TIR-Agent（2026）**和 **OPERA（2026）**：已经覆盖任务识别、工具顺序、reflection、rollback、经验记忆、SFT/RL 工具策略与 planner–tool 联合训练。AgenticIR 明确将调度解释为处理顺序上的搜索。把相同框架换成 3D 工具，不足以单独形成方法创新。[RestoreAgent](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c78f639424b8d89ceb4f2efbb4dfe4f4-Abstract.html)；[AgenticIR](https://arxiv.org/html/2410.17809v1)；[TIR-Agent](https://arxiv.org/abs/2603.27742)；[OPERA](https://arxiv.org/abs/2605.22104)。

## 7. 验证器与 benchmark：可计算不等于可验证

### 7.1 内部一致性并不能证明恢复正确

如果相对位姿由同一组绝对位姿导出，T_ij = T_i⁻¹ T_j，则 T_ij T_jk T_ki = I 恒成立，即使绝对位姿错误。这样的 cycle consistency 是代数自洽，不是外部证据。用于拟合的重投影残差、多次共享偏差的预测，以及错误几何的自渲染，也可能不能区分真实与错误解释。

**SysCON3D / Can These Views Be One Scene?（2026-05-18）**研究神经重建支持的无参考指标如何对不一致图像产生伪几何支持。其 classical verification 也不是普适真值：真实低纹理、反光、低重叠场景仍可能失败。验证器失败与输入本身不可恢复，应明确区分。[全文](https://arxiv.org/html/2605.18754v1)。

重新处理同一批图像仍可能挖掘尚未计算的真实证据，不要求每次都新增传感器数据。但必须记录证据共享的图像、特征、初始化和先验，避免把相关输出当作多份独立支持。若所有可用观测对两个几何解释都等价，更多推理 token 不能凭空消除不可辨识性。

### 7.2 数据集与参考来源决定结论范围

**RealX3D（2025-12-29）**提供退化/干净多视图和激光几何，camera reference 则来自干净图像的 COLMAP 与激光配准。它分别研究已知位姿的优化式 NVS 和仅退化输入的 pose/geometry；motion blur 中还含渲染积分合成，不能把所有退化都叫物理实拍。恢复任务必须明确哪些参考仅供离线评价。[全文](https://arxiv.org/html/2512.23437v1)。

**MVM-IOD（2026-06-15）**有工业物体扫描与标定参考，揭示采集分布变化、输入旋转/宽高比影响。它支持可靠性问题存在，但特定预处理收益不是通用恢复机制，各方法输入量和分辨率差异也限制运行时间的直接比较。[全文](https://arxiv.org/html/2606.16638v1)。

**When Wider Views Fail（2026-09-21）**展示全局组装失败，但只有两个 Google Earth Studio 场景，reference 是更多图像的 VGGT-Ω 输出。其分数是对 pseudo-reference 的符合程度，不能当独立几何真值。[全文](https://arxiv.org/html/2609.24839v1)。

ETH3D 和 DTU 可提供几何参考，但原有 MVS protocol 不自动定义未知外参的 recovery 任务；Map-free 的 single-frame / up-to-9-frame 输入协议不能用整段 SfM 偷换。CO3D、Tanks and Temples 仍需按具体版本审计 reference 与 track，本轮没有补写未核实细节。[ETH3D](https://www.eth3d.net/overview)；[DTU](https://roboimagedata.compute.dtu.dk/?page_id=36)；[Map-free 官方仓库](https://github.com/nianticlabs/map-free-reloc)。

研究结论至少应区分正确相机与已注册相机、局部残差与全局结构、保留下来的样本精度与整体覆盖。不能通过删掉困难相机或对断开分量分别自由对齐，掩盖恢复失败。成本应包括诊断、未采用的分支、失败执行和验证，而不只统计最终成功路径。

## 8. 候选 research questions：尚未确认为空白

以下五个问题是从上述边界推导的候选，不构成方向排名。每一个都需要进一步缩窄到具体失败分布和信息条件；不宜同时堆成五个模块。

### RQ1：错误但自洽的重建，能否通过有成本的诊断计算区分？

**问题。** 当相似外观、重复结构或先验偏差产生多个可解释当前日志的几何假设时，系统应计算哪一项额外证据，才能决定修复哪部分？重点不是“识别失败类别”，而是判别会导致不同恢复动作的竞争解释。

**可能的技术对象。** 与相机、对应关系和结构相联系的多假设状态；操作后的证据更新；诊断动作对后续几何决策的价值估计。例如某次重新验证暂时不改善重建，却可能排除一个错误全局拼接。

**已有覆盖与必要区别。** 重复结构恢复、Doppelgangers++、Progressive SfM 和 Subgraph Pruning 已检测冲突并调用局部求解；DAD/dual control 已研究信息选择。新机制必须具体处理相关几何观测、错误共识和动作副作用，不能只改写成 POMDP。

**可证伪点。** 如果当前静态统计或现有筛边模型已足以区分这些状态，历史和多假设没有额外信息；如果现有图像根本不含区分证据，重新调工具也无法解决。此时应承认不可辨识，或明确进入新增观测的另一任务协议。

**Agent 的地位。** 序贯控制可能合理，LLM 并非必需。只有异质视觉语义证据或变化的工具能力确实需要它时，才有进一步理由使用 VLM/LLM。

### RQ2：能否选择“对最终恢复有价值”的诊断，而不是最高 uncertainty？

**问题。** 在有限计算预算下，某个操作的价值既包括立即修复，也包括它会怎样改变后续操作。高 uncertainty 不一定影响最终相机质量；一个昂贵诊断也可能产生很多无用信息。

**可能的技术对象。** 面向最终几何损失、覆盖及执行成本的诊断/修复价值模型；显式区分信息收益和当前误差下降，并处理先诊断后修复的延迟信用分配。真实几何在在线阶段不可见，只能使用合法观测估计价值。

**已有覆盖与必要区别。** GraphMatch、图网络流选择、GeoCond、Dynamic Algorithm Configuration 和 DAD 都是直接参照。必须找出静态排序、单步 gate 或短搜索无法表达的依赖，而不能把学习动作选择本身当创新。

**可证伪点。** 如果动作顺序几乎无影响，或小型数值模型加当前状态即可获得同样选择，长时域 Agent 不必要。如果收益来自增加工具调用总量，则没有证明价值估计更好。

**Oracle 边界。** 单步 GT-greedy oracle 会低估纯诊断动作；有限搜索 oracle 也不是多步严格上界。有 oracle 提升空间，不代表可见信息足以学出对应选择。

### RQ3：无在线 GT 时，何时可以接受修复、回滚或拒绝判断？

**问题。** 系统如何判断“更自洽”是否真的对应“更正确”，并识别自己无法认证的情况？验收对象必须明确为 pose、表面几何、覆盖或某个风险事件，不能统称 confidence。

**可能的技术对象。** 考虑坐标规范和证据依赖的几何验证量；与实际误修复事件对齐的风险校准；将接受、回滚与 abstention 绑定到可说明的适用条件。在线 GT-free 与训练/校准是否使用 GT 必须分开声明。

**已有覆盖与必要区别。** Trust3R、GeoCond、ReVar3R、SysCON3D 和 conformal stopping 已分别涉及可靠性、门控、扰动一致性、场景验证及停止。不能重复“多跑几次看分歧”，必须指出能检测哪一类此前稳定而错误的输出。

**可证伪点。** 如果验证器只是在低纹理困难输入上拒绝一切，或通过丢掉困难相机提高分数，则没有改善恢复。如果所谓保证依赖不成立的独立性或跨域校准假设，应缩小主张。

**Agent 的地位。** 这可能首先是一篇几何可靠性论文；Agent 是应用场景，不必成为方法主体。

### RQ4：局部修图如何同时控制错误约束与保持全局可恢复性？

**问题。** 删除可疑边可能清除 outlier，也可能切断弱连接、丢掉相机；增加边可能补充约束，也可能引入重复结构假共识。局部分类准确率与最终全局重建目标并不等价。

**可能的技术对象。** 将约束可靠性、几何退化、覆盖和误修复代价联合表达的图决策目标；根据操作后的全局可恢复性更新策略。图连通只是必要背景之一，不能代替可观测性和基线/视差条件。

**已有覆盖与必要区别。** 2018 View-graph Selection、TC-SfM、LFOE、VGPA 与 Progressive SfM 已处理其中相当部分。必须给出这些目标未能表达的具体冲突，以及学习或优化机制如何解决，不能只是增加 connectivity penalty。

**可证伪点。** 如果已有全局选图或鲁棒求解器在相同证据下即可化解冲突，则外层 Agent 没有独立价值；如果精度提高完全靠丢弃难视角，应视为覆盖权衡而非成功恢复。

### RQ5：VLM 的语义线索能否转化为可验证的几何约束？

**问题。** 当数值统计无法区分两个解释时，视觉语言模型是否能提供现有视觉/图模型没有有效利用的关系线索，并将其转成可执行、可检验的约束？关键是约束能被真实图像支持或驳回，而不是文本解释听起来合理。

**可能的技术对象。** 从语义关系生成候选几何假设，再利用对应点、遮挡、跨视角证据等验证；明确语义先验与测量证据的不同角色，并保留错误建议被拒绝的路径。

**已有覆盖与必要区别。** LFOE、Doppelgangers++、MVDG 已使用高层视觉上下文，SpatialClaw 已使用程序化几何分析。需要说明额外能力来自何种预训练知识或可组合关系，而不是 VLM 获得了更多图像、更强 matcher 或更长预算。

**可证伪点。** 如果非语言视觉模型或图网络使用相同输入即可获得同等收益，语言部分不是必要贡献；如果线索无法映射到可检验约束，只能用于主观 judge，则与客观几何恢复主张不匹配。

## 9. 从系统构想到研究贡献：需要明确的论点

一个候选机制应能补完这句话：“在某类真实失败中，已有方法因缺少某种信息/表示/目标而失败；我们通过具体机制获得或利用它，因此改变了一个可客观衡量的几何结果。”如果只能补成“Agent 会调用更多工具”，还没有完成定位。

同工具、同输入、同预算的通用 coding agent 是重要参照，但不应成为唯一参照。传统几何算法、小型非语言控制器、一次性配置预测与短搜索，分别检验新机制是否需要语言、历史和长时域决策。这些是贡献归因的逻辑要求，不是在此启动实验。

系统与 benchmark 也可以成为有效贡献，前提是建立可复用任务、真实失败分布或可靠评价，并产生新的能力或知识。既不应以工具数量撑大论文，也不应以“必须新 backbone”否定清楚的小型算法或系统贡献。

## 10. 投稿边界与版本处理

CVPR 2027 当前 CFP 将 2026-09-15 之后上线的工作一般视为同期工作，明确不会基于与同期工作的比较而拒稿；仍应引用、讨论，并在可行范围内做实验比较。实际影响投稿的工作必须承认；arXiv 与正式发表适用同一规则，日期边界由 AC 判断。ARR/ACL 的三个月规则同样免除的是详细实验/深入比较义务，而非可以完全忽略相关工作。[CVPR 2027 官方 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)；[ACL 官方 Citation 政策](https://www.aclweb.org/adminwiki/index.php/ACL_Policies_for_Review_and_Citation)。

政策保护与研究价值分开判断。近期论文不必自动导致改题，但其公开方法仍可能改变技术定位。Think3D 的首版是 1 月，Agentic Real2Sim 的首版是 7 月；不能以 10 月更新日期把整项工作重新归为同期新工作。新版本新增内容也不能追溯为首版已经具备。

## 11. 优先精读路径与尚未覆盖的边界

先读恢复机制：2014 reconstruction correction → Progressive SfM → Subgraph Pruning。关注哪些错误可发现、哪些操作可撤销、何时真正执行重建获取证据。

再读现代几何方法：LFOE → MP-SfM → GLUEMAP → VGPA。关注增加的信息、状态表示、先验误差处理与覆盖，而非只比较 pipeline 方框。

再读可靠性：GeoCond → ReVar3R → SysCON3D。关注训练标签、在线可见信息、坐标规范、稳定偏差与拒绝条件。

最后对照 Agent：AgenticIR → SpatialClaw → Agentic Real2Sim → LiteReality-Agent。关注控制器与底层工具是否被分离比较、反馈来自何处、成功究竟被怎样定义。

本文依据论文、正式会议、作者项目及代码组织结论；没有复现性能，也不把摘要级核查当全文审计。重点方法的论点与局限来自方法/评测阅读；部分背景项已明确标注摘要或项目级依据。未独立确认正式发表状态的新近工作按预印本处理。

覆盖仍不等于穷尽：SLAM recovery、动态/非刚性 SfM、rolling shutter、自标定、全部主动 SLAM 与 metareasoning 理论尚需在候选问题收窄后补查。没有找到同名论文不是空白证明；有相似框图也不代表研究问题完全重复。

**当前待解决的核心判断：是否存在一种现有方法难以处理、却有合法可获得的额外证据可以区分的几何失败；若存在，什么机制能把这种证据转成更可靠的恢复决策。** 在这一步明确之前，不以工具开发规模或 oracle 提升替代选题论证。

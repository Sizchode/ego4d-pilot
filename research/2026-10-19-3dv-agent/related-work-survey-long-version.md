# 3DV Reconstruction + Agent：Related Work Survey

文献截点：2026-10-19｜Benchmarks、Baselines、Taxonomy 与拟议贡献

本文围绕真实多视角重建的失败恢复组织相关工作：在给定图像、初始重建状态与计算预算下，选择诊断和修复操作，改善相机估计与几何。综述先确定现有 benchmark 能验证什么，再建立 baseline 层级与方法 taxonomy，最后将我们的拟议贡献对应到具体 gap。下文不报告尚未获得的实验结果。

## 1. Benchmarks：现有评测资源与 recovery protocol

### 1.1 几何与位姿：哪些 benchmark 能承担主要证据

**应以位姿和几何恢复为主评价，以渲染为辅助。** 数据集有官方 split，不代表已经定义了 recovery 任务；需要另行明确初始失败、可见信息、动作权限及预算。下表中的“用于本题”是拟议定位，不代表这些数据集的官方任务就是 Agent recovery。

| Benchmark / 数据资源 | 原有任务与 reference | 主要评价量 | 用于本题的位置与限制 |
|---|---|---|---|
| [ETH3D](https://www.eth3d.net/overview) | 多视图几何；激光参考 | 官方 accuracy、completeness、F-score | 场景级几何主证据；reference 相机与激光表面应分别审计，不能假设逐帧外部真值 |
| [DTU MVS](https://roboimagedata.compute.dtu.dk/?page_id=36) | 标定多视图；结构光表面、标定相机 | accuracy、completeness；遵循具体评测脚本的聚合 | 受控几何与位姿评价；未知外参任务须隐藏参考外参，并单列内参可见条件 |
| [Tanks and Temples](https://jaesik.info/publications/data/17_siggraph.pdf) | 大场景表面重建；激光扫描 | precision、recall、F-score | 大场景几何泛化；官方表面分数不自动等价相机恢复率 |
| [IMB / PhotoTourism](https://github.com/ubc-vision/image-matching-benchmark) | 宽基线 matching / 小图像集 SfM；大规模 COLMAP pseudo-GT | 对应版本的相对 pose mAA | 宽基线相机与匹配比较；须固定 benchmark 版本，不能混用不同年份 IMC 定义 |
| [CO3D](https://openaccess.thecvf.com/content/ICCV2021/papers/Reizenstein_Common_Objects_in_3D_Large-Scale_Learning_and_Evaluation_of_Real-Life_ICCV_2021_paper.pdf) | 物体视频；COLMAP 相机/点云及质量筛选 | 选定协议的 pose error / AUC；NVS 为辅助 | 多类别跨对象泛化；reference 非独立扫描，质量筛选也影响自然失败分布 |

DTU 官方说明相机由标定工具得到，表面来自结构光测量；这些 reference 仍有测量误差。IMB 明确称其参考为 pseudo-ground truth，来自 exhaustive COLMAP；CO3D 原论文明确用 COLMAP 做 camera tracking 和 dense reconstruction。因此，后两者适合比较与高质量 SfM 参考的一致性，但不能单独证明恢复了同一参考流程无法识别的系统性错误。[DTU 采集与标定](https://roboimagedata.compute.dtu.dk/)；[IMB 官方协议](https://github.com/ubc-vision/image-matching-benchmark)；[CO3D 原论文](https://arxiv.org/html/2109.00512v1)。

**Benchmark 组合应按证据互补，而非堆数量。** ETH3D/DTU/Tanks and Temples 提供几何侧参考；PhotoTourism/CO3D 提供真实外观和类别变化；歧义及退化数据用于失败分层。先确定目标失败，再从这些资源中构建相互补充的评价，避免把所有数据混成一个平均分。

### 1.2 失败、退化与验证：补充 benchmark 和诊断测试

| 资源 | 所针对的问题 | 可提供的证据 | 与 recovery 的差距 |
|---|---|---|---|
| [RealX3D](https://arxiv.org/html/2512.23437v1) | 低质多视图与重建退化 | 激光几何、干净/退化图像；camera reference 由干净图像估计 | 已知位姿 NVS 与未知位姿恢复是不同 track；干净参考不可泄漏到在线控制 |
| [MVM-IOD](https://arxiv.org/html/2606.16638v1) | 工业对象与采集分布变化 | 独立扫描及标定参考；输入变化下的失败 | 是 OOD / 几何资源，不自带序贯动作协议；输入量需统一 |
| [Doppelgangers++ / VisymScenes](https://arxiv.org/html/2412.05826v2) | 相似结构导致错误匹配 | 视觉歧义、pair 判断及下游 SfM | pair 标签不等价全局 pose 真值；应区分筛边效果与端到端恢复 |
| [SysCON3D](https://arxiv.org/html/2605.18754v1) | 不一致视图的伪几何支持 | 检验 verifier 对错误输入的敏感性 | 一致性任务不能直接代替相机恢复评价 |
| [When Wider Views Fail](https://arxiv.org/html/2609.24839v1) | 视角跨度与全局组装失败 | 有限场景中的现象与诊断 | 两个合成采集场景、模型 pseudo-reference；作为补充现象，不作为主 GT benchmark |

Map-free 的官方 single-frame / up-to-9-frame 约束适合相对定位扩展验证，不适合直接把全序列 SfM 放入其官方排名。E3VS、空间 QA 和生成场景 benchmark 则验证不同输出，不作为本文相机/几何恢复主指标。[Map-free](https://github.com/nianticlabs/map-free-reloc)；[E3VS](https://arxiv.org/abs/2604.17969)。

### 1.3 我们需要补充的 evaluation protocol

**主任务边界。** 暂以固定图像集、未知外参、可声明的初始 reconstruction state、同一在线工具权限为讨论对象。可选内参是否已知应分 track；请求新视角属于额外采集设置，应与固定输入的恢复分开。系统输出是修复后的 cameras / geometry，以及接受、回滚或无法判断的决定。

**初始状态与失败分布。** 所有方法在相同初始状态开始；从头重建的强系统另作端到端比较，不能与修复增量混报。自然失败是目标对象，受控删边/污染只用于机制分析；应覆盖多个初始化 backend，避免只选择某个弱默认配置容易失败的样本。训练、校准与测试按场景/对象划分，不能把同一场景的不同损坏版本分到两侧。

**报告层级。** 全集上的最终质量与开销说明系统实用价值；预先定义的失败子集上报告恢复率，说明 recovery 能力；原本成功的子集上报告 regression / false-repair rate，说明误修复风险。成功门槛由验证协议预先确定，不在测试结果上反向挑选。

**指标层级。** 位姿：rotation / translation-direction error、适用时的 mAA/AUC、正确恢复的 camera coverage；有可靠尺度/轨迹时再使用 ATE/RPE。几何：保留各 benchmark 官方 accuracy/completeness/F-score。系统：场景成功率、误修复率、风险–覆盖曲线、完整 wall-clock/GPU/模型调用开销及 quality–cost curve。PSNR/SSIM/LPIPS 仅是渲染辅助证据。

**对齐与成本。** 无尺度重建采用预先定义的整体 gauge 对齐；不能把每个断开分量分别自由对齐后当作全局成功。registered-image ratio 应与 pose correctness 联报，不能用删掉难视角后的低平均误差代替恢复。预算计入诊断、失败调用、候选分支和验证；共享缓存与预计算的权限保持一致。

这一 protocol 的潜在贡献是把已有数据转成可复用、可比较的 recovery evaluation，而非声称又创造了一套图像数据。只有它提供现有评测无法回答的问题，并有可复现状态、评价与多方法证据时，才足以成为 benchmark contribution。

## 2. Baselines：比较对象、优先级与归因

### 2.1 完整重建系统：证明最终结果有竞争力

下列是需要优先核对的 baseline 家族，不是已经复现的排行榜。选择应匹配任务规模、相机条件、可用实现和输入协议；不能用各论文不同数据/预算的分数直接横向排名。

| Baseline 家族 | 代表方法 | 对照目的 | 应控制的条件 |
|---|---|---|---|
| 经典增量 / 全局 SfM | [COLMAP](https://colmap.github.io/)、[GLOMAP](https://arxiv.org/html/2407.20219v1) | 恢复收益是否超过成熟后端 | matcher、候选图、内参、BA；固定论文/代码版本 |
| 强前馈及其优化版本 | [VGGT](https://github.com/facebookresearch/vggt)、VGGT + BA | 是否只是获得更强几何初始化 | raw 与 BA 分列，固定 checkpoint、帧数及分辨率 |
| 完整 learned / hybrid SfM | [MASt3R-SfM](https://arxiv.org/abs/2409.19152)、[MP-SfM](https://arxiv.org/html/2504.20040v1)、[GLUEMAP](https://arxiv.org/html/2605.26103v2) | 超出简单模型 fallback 的系统价值 | 保留各自完整推理链；统一合法输入并记录全部成本 |
| 学习修图 / 消歧 | [LFOE](https://openaccess.thecvf.com/content/CVPR2025/papers/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.pdf)、[Doppelgangers++](https://arxiv.org/html/2412.05826v2)、[VGPA](https://arxiv.org/html/2609.09491v1) | 相机图或对应关系机制是否新增能力 | 能兼容时固定后端；不兼容时按完整系统比较 |
| 执行驱动的传统恢复 | [Progressive SfM](https://arxiv.org/html/1803.07349v1)、[Subgraph Pruning](https://arxiv.org/html/2608.22054v1) | 多步反馈是否超出已有恢复 | progressive 输入与固定输入协议需匹配或明确标为改编 |

VGGT 官方已经提供带/不带 BA 的导出流程，不能用其 raw 输出代表全部 baseline 能力。COLMAP/GLOMAP 也应区分论文方法、代码版本与当前集成实现，避免把实现更新当作两套独立技术的证据。[VGGT 官方说明](https://github.com/facebookresearch/vggt)。

**主表至少应覆盖不同路线的强代表。** 原始 COLMAP、固定 retry 和 exhaustive matching 可以作为参照，但不足以支撑“优于现有恢复方法”。反过来，exhaustive matching 也不是严格 oracle：更多候选边可能带来歧义和 outliers。

### 2.2 同工具控制器：证明贡献来自决策机制

这些是本研究需要定义的 controlled baselines，不冒充已有同名论文；它们与上面的完整系统比较承担不同任务。

| 控制器对照 | 保持相同的内容 | 检验的贡献 |
|---|---|---|
| 固定 retry / 数值规则控制器 | 工具、输入、预算、初始状态 | 基础调度是否已足够；规则是基线，不是拟议方法 |
| 一次性配置或动作序列预测 | 初始观测和可调用操作 | 执行反馈是否比初始状态有增量 |
| 小型非语言模型 / DAC 控制器 | 数值和图状态、动作权限 | 是否真的需要语言模型，还是普通 algorithm control |
| 短时域搜索 / 贪心选择 | 同一 proxy verifier 与完整执行成本 | 是否需要长期决策和历史表示 |
| 通用 coding agent | 同样图像、工具说明、预算、可见历史 | 是否超出通用工具编排；模型版本与提示词需记录 |
| 拟议方法的去机制版本 | 同一 backbone 和执行工具 | 分离证据表示、动作价值、验收机制的增量 |

DAC 的思想已有成熟文献；图像恢复 Agent 的 rollback、reflection、memory 和工具策略训练也已有先例。因此“用了 RL”或“多轮比单轮好”本身不足以定位贡献。[Dynamic Algorithm Configuration](https://arxiv.org/abs/2205.13881)；[AgenticIR](https://arxiv.org/html/2410.17809v1)；[TIR-Agent](https://arxiv.org/abs/2603.27742)。

### 2.3 诊断、验收与 oracle：检验判断是否可信

**诊断/验收对照。** 原始 residual/confidence、独立 pairwise consistency、扰动分歧，以及适用任务下的 GeoCond、ReVar3R、Trust3R、SysCON3D。它们分别预测不同对象，不能把像素 uncertainty、相机 reliability 和 scene consistency 当作同一输出强行排名；需要映射到明确的错误事件，并承认改编方法与原论文不同。[GeoCond](https://arxiv.org/html/2609.18465v1)；[ReVar3R](https://arxiv.org/html/2610.07883v1)；[Trust3R](https://arxiv.org/html/2605.19539v1)；[SysCON3D](https://arxiv.org/html/2605.18754v1)。

**Oracle 分层。** 单步 GT-greedy 用于判断即时动作空间；有明确深度/预算的搜索 oracle 用于分析序列潜力；二者都使用在线不可见信息，只是分析参照。单步 oracle 可能低估纯诊断动作，有限搜索也不是全局最优上界。不能把 oracle headroom 当作实际可学习或可部署的证明。

**相邻 Agent 的位置。** Real2Sim、LiteReality-Agent、FACT、RSR 属于必须讨论的 related work；其相机条件、输出和工具不同，不宜把原论文分数塞进同一 recovery baseline 表。若借用控制架构，要标为 adapted baseline，并保持同工具权限。

## 3. Taxonomy：其他工作按解决的问题分类

### 3.1 分类轴与总体定位

采用四条轴：改变什么对象（数据、约束、位姿、显式场景或答案）；新证据从哪里来（图像计算、先验、外部观测或自生成反馈）；控制如何形成（优化、学习预测、序贯策略或语言编排）；最终验证什么（几何、渲染、QA 或物理任务）。同一论文可以跨类，不强行互斥。

| 类别 | 代表工作 | 主要输出 / 反馈 | 与我们拟议定位的关系 |
|---|---|---|---|
| 图构建与选图 | GraphMatch、View-graph Selection、2021 Construction | 可信边与子图；几何验证 | 提供补边/选边机制，非所有异质操作的控制 |
| 鲁棒估计与结构恢复 | 1DSfM、2014 修复、Progressive SfM、Subgraph Pruning | cameras / structure；冲突与局部重建 | 最直接传统前例；闭环恢复本身已存在 |
| 学习图推断 / 消歧 | LFOE、Doppelgangers++、VGPA、MVDG | edge / pose / covisibility；视觉几何训练 | 不使用语言也能融合复杂图像信息 |
| 神经几何与优化融合 | MASt3R-SfM、MP-SfM、GLUEMAP | cameras / dense geometry；先验与 BA | 增加约束的强系统，限制简单 fallback 空间 |
| 可靠性与计算选择 | LoRA3D、Trust3R、GeoCond、ReVar3R | confidence、gate、refinement | 已覆盖部分诊断与验收；目标层级不同 |
| 主动采集 | AREA3D、Active3D | 新视角与更充分重建；采集反馈 | 改变输入权限，应独立于固定图像恢复 |
| 空间 QA / 工具推理 | Think3D、SpatialClaw、Skill-3D、MAG-3D | 答案；渲染和程序证据 | 贡献常在推理接口，非提高底层几何 |
| 真实场景 / real-to-sim Agent | Real2Sim、LiteReality-Agent、FACT、RSR | 对象/场景资产和仿真；critics/物理反馈 | 近邻闭环已存在；未知相机恢复不是统一目标 |
| 生成编辑与视觉 restoration | SAGE、SceneSmith、MUSE、AgenticIR、OPERA | 场景或修复图像；视觉/任务反馈 | 限制通用框架 novelty，不能视为同任务 baseline |

以下按机制展开相关工作。重点是每一类已经解决了什么，以及与本研究需要解决的问题具体差在哪里；不以“是否含 Agent 一词”划分新旧。


### 3.2 鲁棒几何、图构建与传统恢复

#### 3.2.1 错误结构恢复已经是明确研究对象

**Correcting for Duplicate Scene Structure in Sparse 3D Reconstruction（ECCV 2014）**直接研究失败重建中的错误检测与纠正。**Recovering Correct Reconstructions from Indistinguishable Geometry（3DV 2014）**利用几何共现定位致错结构，拆成子模型后重新合并。它们针对重复/不可区分几何，并不等于解决所有失败，但已覆盖“先诊断、再恢复”的基本结构。此处依据原始摘要和机构页面；前者全文访问未成功，不据此推断实现细节。[ECCV 论文](https://www.cs.unc.edu/~jheinly/publications/eccv2014-heinly.pdf)；[3DV 机构记录](https://researchwith.stevens.edu/en/publications/recovering-correct-reconstructions-from-indistinguishable-geometr/)。

**Progressive Structure from Motion（ECCV 2018）**是尤其重要的近邻。它随输入和图拓扑变化复查局部结果，进行重聚类、结构转移、拆分、合并和 reset/re-estimate；全局保留的约束使后续证据有机会纠正早期决策。其输入协议是 progressive，而非任意固定图像集上的通用控制器，但“反馈、可逆性、积累证据后重判”已存在。潜在增量必须落在新的状态表示、目标或决策机制上。[全文](https://arxiv.org/html/1803.07349v1)。

#### 3.2.2 图构建与选择并非只有静态 retrieval

**GraphMatch（3DV 2017）**通过 sample-and-propagate 提出新图像对，几何验证后更新图邻域并继续搜索。它已经把当前计算结果用于后续匹配，主要目标是高效发现有效边，而非解释复杂失败。[全文](https://arxiv.org/html/1710.01602v1)。

**View-graph Selection Framework for SfM（ECCV 2018）**使用 minimum-cost network flow，结合图与几何/context priors 选择子图，兼顾歧义、效率和准确率。预算式图选择不能被概括为简单阈值规则。[正式论文](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Rajvi_Shah_View-graph_Selection_Framework_ECCV_2018_paper.pdf)。

**View-graph Construction Framework for Robust and Efficient SfM（Pattern Recognition 2021）**从已验证的生成树出发，利用局部重建、相机注册产生新的相对几何并迭代扩图。因此“执行局部重建获得证据，再更新图”也已有先例。[出版社原文](https://www.sciencedirect.com/science/article/pii/S003132032030515X)。

#### 3.2.3 鲁棒估计与近期恢复近邻

**1DSfM（ECCV 2014）**从一维投影中的顺序冲突识别异常相对平移；**COLMAP（CVPR 2016）**本身包含增量注册、三角化、过滤和 BA；**GLOMAP（ECCV 2024）**通过旋转平均、相机与点联合 global positioning 和 BA 求解。它们都不是等待外部 Agent 才处理错误的“正常流程”。[1DSfM](https://research.cs.cornell.edu/1dsfm/)；[COLMAP](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf)；[GLOMAP](https://arxiv.org/html/2407.20219v1)。

**TC-SfM（2022 预印本）**利用 track-community/context 识别歧义部分，分块重建并进行一致性合并。作者指出弱连接可能导致部分相机被隔离或丢失，提示“删掉不可靠约束”与“保持可恢复性”存在冲突。[全文](https://arxiv.org/html/2206.05866v1)。

**Robust Global SfM via View Graph Pruning（2026-08-22 预印本）**递归划分图并实际运行子图 GLOMAP，再通过一致性和跨块 PnP/RANSAC 验证剪边。它已经是执行驱动的诊断，而非只读取静态统计。其算子顺序和门槛预定义，未研究异质诊断计算的长期选择价值；这个区别仅针对该文，不是全领域空白证明。[全文](https://arxiv.org/html/2608.22054v1)。

### 3.3 学习图推断与神经几何融合

#### 3.3.1 视觉信息与图学习已有直接应用

**Doppelgangers（ICCV 2023）及 Doppelgangers++（CVPR 2025）**区分外观相似、实际属于不同三维表面的图像。后者使用 MASt3R 的 3D-aware features 和 Transformer 分类器，并集成到 SfM。视觉高层线索识别假匹配，并非 VLM 独有能力。[Doppelgangers](https://openaccess.thecvf.com/content/ICCV2023/html/Cai_Doppelgangers_Learning_to_Disambiguate_Images_of_Similar_Structures_ICCV_2023_paper.html)；[Doppelgangers++](https://arxiv.org/html/2412.05826v2)。

**Learning to Filter Outlier Edges in Global SfM / LFOE（CVPR 2025）**把原图的边变为 line graph 节点，以相对几何、预估旋转和 DINOv2 特征学习过滤。它既有跨数据评价，也有 oracle filter。该 oracle 只针对特定删边准则，作者指出处理退化配置仍可能超过它，因此不能把它当作全 SfM 的严格上界。[正式论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.pdf)。

**VGPA（ECCV 2026；arXiv 2026-09-08）**通过 edge-conditioned GNN 从 noisy view graphs 估计全局相机，推理时还有场景一致性 adaptation，并可重新纳入被删相机。“学习图处理 + test-time adaptation + 相机恢复”已经有组合；它并非纯一次前向过程。[全文](https://arxiv.org/html/2609.09491v1)。

**MVDG（2026-10-01 预印本）**使用 VGGT 多视图特征消歧，并通过 confidence-based early exit 分配计算。其 early-exit confidence 与最终深层输出的一致性有关，不自动等于真实几何正确率。多视图上下文和自适应计算也需要与此类方法比较。[全文](https://arxiv.org/html/2610.01098v1)。

#### 3.3.2 强 hybrid reconstruction 会改变恢复空间

**MASt3R-SfM（3DV 2025）**将 learned local geometry/matches 与全局对齐结合，是完整重建路线，而不只是可调用的初始化工具。本处依据原始摘要及作者页面。[论文记录](https://arxiv.org/abs/2409.19152)。

**MP-SfM（CVPR 2025）**融合深度、法向先验及不确定性，使用 lifting、depth consistency、扩展 BA 与先验细化，针对低视差、低重叠等困难。其启发是：部分失败来自约束不足，需要增加并正确融合信息；换一个调度器未必能解决。先验 uncertainty 也需要校准。[全文](https://arxiv.org/html/2504.20040v1)。

**GLUEMAP（CVPR 2026）**将局部 feed-forward stars、motion averaging、真实和虚拟 tracks 的 augmented BA 结合起来。这已经覆盖大量“传统 SfM 失败时借助前馈几何”的设想。其剩余限制包括对局部模型质量的依赖，不能因此推导通用恢复已解决。[全文](https://arxiv.org/html/2605.26103v2)。

相邻进展还有 **VGGSfM（CVPR 2024）**的可微 tracks/camera/BA、**G-MASt3R-SfM（2026）**的图剪枝和多阶段优化，以及 **Glob3R（2026）**的跨窗口 tracks 与全局优化。后两者在本轮主要用于补充方法版图，不据摘要宣称性能优劣。[VGGSfM](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_VGGSfM_Visual_Geometry_Grounded_Deep_Structure_From_Motion_CVPR_2024_paper.pdf)；[G-MASt3R-SfM](https://arxiv.org/html/2606.22856v1)；[Glob3R](https://arxiv.org/abs/2607.09225)。

### 3.4 可靠性估计与序贯算法控制

#### 3.4.1 从模型置信度到决策可靠性，仍有距离

**LoRA3D（ICLR 2025）**使用鲁棒点图对齐、置信度重估与筛选伪标签，再做场景 LoRA 适配。重建—验证—修正循环已有具体非语言机制，其价值在几何目标与置信度更新的关系。[全文](https://arxiv.org/html/2412.07746v1)。

**Trust3R（ICML 2026）**在冻结 MASt3R 后增加 evidential head 和 gated residual head，预测点分布并选择性修正。它处理单点 XYZ 的协方差，不显式建模跨像素相关性。局部可靠性不能自动解释为全局相机图失败概率。[全文](https://arxiv.org/html/2605.19539v1)。

**GeoCond（2026-09-16 预印本）**从视差、重叠、基线、旋转和 confidence 等特征预测 pose reliability 与 refinement gate，用于是否 BA、图剪枝、旋转平均和采集停止。uncertainty 可用自监督信号训练，但 BA gate 的训练标签利用 GT pose error 判断优化收益；推理不需要 GT。作者将完整 SfM/SLAM 轨迹验证留作后续。这是“读几何诊断再决定调用优化器”的直接近邻。[全文](https://arxiv.org/html/2609.18465v1)。

**ReVar3R（2026-10-06 预印本）**指出不同坐标规范中的点图方差会混入非误差因素，因此先做鲁棒 Sim(3) 对齐再计算多次预测分歧。核心可无标签，完整融合/校准版本使用标签；作者明确承认无法发现稳定系统偏差。多次预测都同意，仍可能一起错。[全文](https://arxiv.org/html/2610.07883v1)。

**VGGT uncertainty audit（2026-06-15 预印本）**提醒置信度含义取决于输出阶段和实际评价目标，BA 后不能直接沿用优化前的不确定性解释。DUSt3R 官方 confidence loss 的形式也不意味着它就是“相机正确概率”。[VGGT audit](https://arxiv.org/html/2606.16479v1)；[DUSt3R 官方实现](https://github.com/naver/dust3r/blob/main/dust3r/losses.py)。

#### 3.4.2 序贯工具选择有更早的方法基础

**Dynamic Algorithm Configuration（JAIR 2022）**已经研究按执行状态动态改变算法配置。若观测是低维统计、动作只是几个 matcher/BA 配置，首先应考虑 algorithm control，而非默认需要语言模型。[论文](https://arxiv.org/abs/2205.13881)。

**Deep Adaptive Design（ICML 2021）**根据历史 design–observation 学习后续实验选择，以全序列信息增益为目标；**dual control**研究动作既改善系统又提供信息的双重作用。这些思想与“先做诊断，再决定修复”相关，但不能直接套用：SfM 的诊断工具离散、成本不同、观测相关，且常缺乏显式 likelihood。[DAD](https://proceedings.mlr.press/v139/foster21a.html)；[Dual Control and Online Optimal Experimental Design](https://epubs.siam.org/doi/10.1137/16M1069936)。

因此 belief state、POMDP、信息增益或 RL 这些术语本身不是新贡献。必须说明隐藏量、可用证据、坐标规范、动作副作用和几何目标如何使问题不同。软件状态可读，不代表真实错误原因可知；反过来，也不能仅因存在 Agent 就宣布需要部分可观测模型。

### 3.5 主动采集、空间推理与真实场景 Agent

#### 3.5.1 主动视角与空间 QA：改变观察不等于修复估计

**AREA3D（CVPR 2026）**将重建置信度和 VLM 区域指导组成 utility fields，以候选视角排序和覆盖更新选择采集动作。**Active3D（2025 首稿）**结合 uncertainty、规划与局部 refinement。不能说主动重建都假设流程正常，但它们主要研究采集和表示更新，而非错误 correspondence/相机图恢复。[AREA3D](https://arxiv.org/html/2512.05131v2)；[Active3D](https://arxiv.org/html/2511.20050v1)。

**Think3D（首版 2026-01-19；所读 v5）**在重建后通过变换/渲染回答问题，最新版本每个问题最多一次 reconstruction。**SpatialClaw（2026-06）**通过持久 Python 状态组织空间工具推理，并研究接口差别。**Skill-3D（2026-06）**提取 skills/lessons，经 SFT + GRPO 学习工具使用。它们最终评价主要是 QA，并未证明底层 reconstruction 被修复。[Think3D](https://arxiv.org/html/2601.13029v5)；[SpatialClaw](https://arxiv.org/html/2606.13673v1)；[Skill-3D](https://arxiv.org/html/2606.07436v2)。

**Explore until Confident**使用 conformal prediction 研究探索停止；**E3VS-Bench**提供 3DGS 环境的主动搜索任务；**MAG-3D**组织 planner、grounding 和 coding verification。它们说明停止规则、证据获取、程序验证均已有研究，同时必须区分“在重建环境里找答案”和“恢复环境自身”。本段以摘要及作者页面核查为主。[Explore until Confident](https://explore-eqa.github.io/)；[E3VS](https://arxiv.org/abs/2604.17969)；[MAG-3D](https://arxiv.org/abs/2604.09167)。

#### 3.5.2 真实重建与 real-to-sim：最直接的 Agent 近邻

**Agentic Real2Sim（首版 2026-07-21；所读 v1/v5）**使用标定、多视角/深度及机器人轨迹，critic 可重选 mask、关键帧和 tracking 初始化，再根据回放修改。其对象是可回放的交互 episode，不是通用未知相机图。v5 已加入 PSNR、SSIM、ADD-S 和机器人用途，不能说只有 VLM judge；但 replay success 仍有最佳候选和 judge 定义，且 one-prompt 对照缺少相同专用工具，不能据此隔离控制器的贡献。[版本记录](https://arxiv.org/abs/2607.19190)；[v5 全文](https://arxiv.org/html/2607.19190v5)。

**LiteReality-Agent（2026-10-01）**从 RGB-D scan、已知相机、点云与 RoomPlan layout 出发，观察证据、编辑 Room.py，检查碰撞、支撑和 articulation。它主要修复可编辑场景资产；评估使用采集视角，不能当作 held-out novel-view 验证。observe–edit–verify 的真实场景闭环已经存在。[全文](https://arxiv.org/html/2610.01863v1)。

**FACT（首版 2026-09-29；所读 v2）**对 articulated twin 进行定向测量、几何修订、接触/碰撞与动力学拟合。其几何参考包含 image-generated mesh，不能等同独立真实几何 GT。“先测量、再诊断、再修改表示”不能作为独有机制。[全文](https://arxiv.org/html/2609.37067v2)。

**Agentic RSR（首版 2026-10-07；所读 v2）**从视频及前馈重建出发，结合已知机器人 URDF 确定尺度，修改 Blender 场景并做交互检查。其参考相机被冻结；depth feedback 来自先前预测，而非独立传感器真值。它已有重建返工，但不等于恢复相机估计。[全文](https://arxiv.org/html/2610.10479v2)。

**Agentic Building-Aware Satellite GS（2026-09-22）**在有限权重计划中选择配置，依赖候选验证 DSM 指标，只有四个 JAX 场景；不能当作已验证的 GT-free 通用在线恢复。**Harvard SceneAgent**的第一方页面也包含 COLMAP→3DGS 的检查与修正，但本轮未确认独立论文或首次公开日期。[卫星工作](https://arxiv.org/html/2609.25578v1)；[SceneAgent 项目](https://computationalrobotics.seas.harvard.edu/SceneAgent/)。

其他相邻工作包括 **AGILE** 的手物交互重建、**Real2Gym** 的视频到仿真与技能复用、**RoboSimStudio** 的可编辑仿真工具。它们支持拓宽近邻范围，本轮摘要核查不足以判断其所有模块消融。[AGILE](https://agile-hoi.github.io/)；[Real2Gym](https://arxiv.org/abs/2609.37089)；[RoboSimStudio](https://arxiv.org/abs/2610.18835)。

#### 3.5.3 生成编辑与 restoration agents 限制通用框架新颖性

**SAGE、SceneSmith、MUSE**分别强调场景生成及 critics、分层设计和资产物理属性、局部编辑与约束保留。这些任务可以有系统贡献，但不能与从真实图像恢复相机/几何混为一谈。[SAGE](https://arxiv.org/abs/2602.10116)；[SceneSmith](https://arxiv.org/abs/2602.09153)；[MUSE](https://arxiv.org/abs/2606.14168)。

更直接的通用结构近邻是 **RestoreAgent（NeurIPS 2024）**、**AgenticIR（ICLR 2025）**、**TIR-Agent（2026）**和 **OPERA（2026）**：已经覆盖任务识别、工具顺序、reflection、rollback、经验记忆、SFT/RL 工具策略与 planner–tool 联合训练。AgenticIR 明确将调度解释为处理顺序上的搜索。把相同框架换成 3D 工具，不足以单独形成方法创新。[RestoreAgent](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c78f639424b8d89ceb4f2efbb4dfe4f4-Abstract.html)；[AgenticIR](https://arxiv.org/html/2410.17809v1)；[TIR-Agent](https://arxiv.org/abs/2603.27742)；[OPERA](https://arxiv.org/abs/2605.22104)。


## 4. Gap synthesis：已有覆盖与仍需解决的问题

**可以主张的 gap 必须是现有方法边界的交集，而不是没人用过同一个标题。** 下表明确区分文献支持的限制与我们拟议解决的部分；“未统一覆盖”仅指本综述所审阅的方法，不构成全领域排他性证明。

| Gap | 已有方法已经做到 | 尚需解决的具体部分 | 对应拟议贡献 |
|---|---|---|---|
| G1：恢复能力缺少统一归因 | MVS/pose benchmark、退化数据、歧义测试均存在 | 将固定初始状态、合法在线证据、完整成本、恢复与误修复联合评价 | C1：可复用 recovery protocol 与多层 baseline |
| G2：可靠性分数到干预选择之间有缺口 | 学习筛边、局部求解诊断、单步 BA gate 已存在 | 区分会导致不同修复选择的竞争解释，并估计下一次诊断对最终几何的价值 | C2：面向恢复的证据状态与预算决策机制 |
| G3：内部改进未必代表真实恢复 | residual、confidence、perturbation、scene verification 均已有 | 在相关证据和稳定偏差下决定接受/回滚/拒绝，且维持覆盖 | C3：与误修复事件对齐的验收与风险评价 |
| G4：Agent 增益容易与工具能力混淆 | 完整系统比较、通用 Agent 及视觉工具策略已有 | 在相同工具/输入/预算下隔离新几何机制，同时对比强完整重建系统 | 作为 C1–C3 的证据要求，不单列为新算法 |

G2 的直接参照是 GraphMatch、Progressive SfM、Subgraph Pruning、GeoCond 和 DAC；G3 的直接参照是 Trust3R、ReVar3R 与 SysCON3D。差别应落在被推断的量和被优化的目标，而非把已有机制重新命名。

### 4.1 为什么“有工具、有 loop”不足以填 gap

给所有现有算法加工具接口，增加了可调用性，却没有自动增加区分错误解释的信息。多步运行也不天然优于单步：如果当前状态已经包含充分信息，或者动作次序没有影响，复杂 Agent 不必要。高置信和低残差也可能来自错误但自洽的结构。

例如 T_ij = T_i⁻¹ T_j 推出的闭环 T_ij T_jk T_ki = I 对任意绝对位姿都成立；它无法独立认证那些位姿。对同一 backbone 反复采样也可能共享稳定偏差。重新处理原图能够挖掘未计算证据，但在所有可得观测都无法区分两个解释时，应承认不可辨识，不能用语言推理替代缺失信息。

因此，拟议贡献应连接“可见证据 → 几何解释 → 有成本的干预 → 可验证结果”，而不是连接几个软件包。模块可以作为 plugin 交付；其研究内容仍应在移除工具品牌后成立。PixSfM 的 featuremetric refinement 和 LFOE 的图学习正是这种区别。[PixSfM](https://arxiv.org/abs/2108.08291)；[LFOE](https://openaccess.thecvf.com/content/CVPR2025/papers/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.pdf)。

## 5. Our proposed contribution：一条可检验的论文主线

**暂定研究定位：Evidence-driven Recovery of Failed Multi-view Reconstruction。** 在固定图像和受限计算预算下，利用执行产生的几何证据选择诊断与恢复，并控制误修复风险。Agent 是执行机制；论文核心是如何形成、使用和验证证据。以下是拟议贡献陈述，不表示方法或实验已经完成。

### C1. Recovery evaluation protocol：让恢复成为可复现的比较对象

**拟贡献。** 在已有 benchmark 上定义可共享的初始重建状态、信息权限和预算记账，联合评价相机/几何改善、覆盖、失败恢复与原本成功样本的误修复。完整系统 baseline 和同工具控制器 baseline 分开，解决 G1 与 G4。

**区别于现有工作。** 重用 ETH3D、DTU 等图像本身不是新贡献；新增价值必须来自 recovery protocol 能回答普通最终精度排行榜回答不了的问题。若已有 benchmark 已充分提供这些能力，应直接采用，C1 降为评价设置，不硬凑贡献点。

**需要的证据。** 跨初始化方法、跨场景的失败状态可复现；规则不偏向某个控制器；独立几何参考与 SfM pseudo-reference 分开报告；不同 baseline 的比较会揭示有解释力的恢复/成本/风险差异。

### C2. Evidence-conditioned recovery policy：从“选工具”推进到“选证据与干预”

**拟贡献。** 表示与相机、对应关系及结构相联系的竞争解释，记录证据来源与相关性；根据剩余预算和后续几何收益，选择具有区分作用的诊断或直接修复。诊断动作允许暂时不改善几何，但应改变后续选择。这是对应 G2 的主要方法候选。

**区别于现有工作。** 相较学习筛边，不只输出局部 edge 标签；相较单步 gate，不只预测当前是否 BA；相较固定局部重建诊断，不预先固定所有诊断顺序；相较通用 Agent，决策依据需要体现 gauge、图结构、证据依赖和操作副作用。上述差别必须由具体模型/目标实现，不能只写在 prompt 中。

**研究问题与否定条件。** 哪类错误需要历史和竞争解释？哪次额外计算能产生可用于区分它们的证据？如果小型非语言控制器、现有图模型或同预算短搜索即可达到相同结果，则应收缩长时域或 LLM 主张。若额外诊断根本没有信息，策略复杂度也不能创造可恢复性。

**Agent / VLM 的可能作用。** VLM 可以提出语义相关的候选解释，但必须转成可被真实观测支持或驳回的几何约束。LFOE、Doppelgangers++ 和 MVDG 已使用高层视觉信息，不能把“看图”当作语言模型的独占优势。LLM 是否必要由同输入、同工具的非语言对照决定。

### C3. Risk-aware acceptance：让闭环知道何时接受、回滚或拒绝

**拟贡献。** 将验收目标定义为实际误修复事件及覆盖损失，结合可检查的几何证据，形成带校准条件的接受/回滚/拒绝判断。对应 G3，并作为 C2 决策的终止与验证部分，而非再堆一个无关模块。

**区别于现有工作。** 不将 native confidence、优化残差或相同模型分歧直接当作成功概率；不让删除困难相机改善平均误差掩盖覆盖下降。需要说明新增验收量发现何种已有方法难以发现的错误，以及在哪些输入条件下不作保证。

**需要的证据与否定条件。** 验收分数应与独立几何改善或恶化对应，并通过风险–覆盖关系显示拒绝不是简单放弃困难样本。训练/校准可以使用明确声明的 GT，在线不可用；跨域校准假设必须说明。如果只是重新组合已有可靠性分数，应定位为组件适配，不单独声称新可靠性方法。

### 5.1 Contribution–baseline–benchmark 对应关系

| 拟议贡献 | 主要反事实 / baseline | 最相关评测证据 |
|---|---|---|
| C1：统一恢复协议 | 默认流程、强完整重建系统；多种初始 backend | ETH3D/DTU/T&T 几何 + PhotoTourism/CO3D 泛化；恢复/误修复与完整成本 |
| C2：证据条件决策 | 学习筛边、GeoCond 类 gate、DAC、短搜索、同工具 coding agent | 歧义/退化分层；最终 pose/geometry；等预算质量与动作价值分析 |
| C3：风险感知验收 | residual/confidence、独立一致性、ReVar3R/Trust3R 类信号 | 独立 reference、成功样本 regression、risk–coverage、coverage 损失 |

**不能预先承诺的结果。** 当前没有证据支持“首次通用 reconstruction recovery Agent”“必然优于 GLUEMAP”“LLM 必不可少”或“无 GT 保证全局正确”。应在相关方法逐项比较后，将论文主张限制到实际解决的失败条件。C2 是主方法候选；C1 与 C3 是否独立成为贡献，取决于它们是否产生超出协议适配与既有组件的新内容。

### 5.2 可用于 proposal 的定位段落

现有研究分别推进了鲁棒 SfM、学习图处理、神经几何融合与 Agent 工具执行，但这些进展尚未使“如何依据可获得的证据，在预算内决定下一步恢复操作，并判断修复是否可信”成为同一可比较的研究对象。本文拟研究固定观测下的 evidence-driven reconstruction recovery：以明确的初始状态和信息权限组织评价，以几何解释及证据价值驱动诊断和干预，并以相机/几何改善、覆盖与误修复风险检验闭环。目标贡献在于恢复决策与验证机制；标准重建工具是执行基础。

这一定位应以文献对照和后续证据继续收窄。它把我们的主张从“搭一个能修复的工具箱”转为“解决哪一种现有方法尚未处理好的证据与决策问题”，并为每项主张给出对应 benchmark 和 baseline。

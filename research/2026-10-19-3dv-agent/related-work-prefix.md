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

# 3D Agent 文献核查与贡献边界

检索截至：2026-10-19。范围：主动重建、空间推理、真实场景/物体重建与修复、场景生成。本文不制定 pilot，也不声称穷尽全部论文。

## 1. 结论与需要纠正的前提

“真实 reconstruction 的 agentic diagnosis / repair 还基本空白、最近仅有卫星 workflow”已不能成立。下面的 Agentic Real2Sim、LiteReality-Agent、FACT、Agentic RSR，分别覆盖 perception 重试、真实室内场景的观察—编辑—验证、几何/碰撞诊断修复、视频重建的执行反馈。它们没有全部解决通用 SfM view-graph / camera recovery，但足以阻止宽泛的“首次 Agent 基于几何诊断修复重建”表述。

同样，“主动重建方法都假设 pipeline 正常”太笼统。Active3D 已使用局部深度/光度残差指导局部 refinement；AREA3D 关注未充分重建区域。更准确的边界应当是：它们主要改变采集策略及场景表示，不把错误 correspondence、相机图拓扑、注册失败的恢复作为独立评测任务。

最有用的分类不是“有没有 Agent”，而是：**改变数据获取、改变内部估计、改变显式场景程序，还是仅改变观察方式；反馈来自真实新证据、内部一致性、物理执行、GT，还是 VLM judge。** 下文按这些区别审阅。

## 2. 日期、状态、核查深度

日期来自 arXiv submission history；它是可核查的 v1 提交时间，不保证作者此前从未在其他网站公开。最后版本时间不能替代首版时间。CVPR 2027 的同期工作政策由主报告单独核查；此处不把 9–10 月论文自动视为题目终结。

| 工作 | arXiv v1 | 检索时最新版本 | 核查状态 / 深度 | 主要对象 |
|---|---|---|---|---|
| [Explore until Confident](https://arxiv.org/abs/2403.15941) | 2024-03-23 | v3 / 2024-07-07 | 作者项目页与摘要；非本轮全文精读 | EQA 探索和停止 |
| [Active3D](https://arxiv.org/abs/2511.20050) | 2025-11-25 | v1 | 方法选读；[作者页](https://liyan.me/news.html)报告 AAAI 2026 oral | 不确定性驱动主动重建 |
| [AREA3D](https://arxiv.org/abs/2512.05131) | 2025-11-28 | v2 / 2026-07-30 | 方法与评测精读；[CVPR 2026 正式记录](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_AREA3D_Active_Reconstruction_Agent_with_Unified_Feed-Forward_3D_Perception_and_CVPR_2026_paper.html) | 双场引导 NBV |
| [Think3D](https://arxiv.org/abs/2601.13029) | 2026-01-19 | v5 / 2026-10-09 | 最新方法精读；本轮不确认录用 venue | 重建后的视角操作与 QA |
| [AGILE](https://arxiv.org/abs/2602.04672) | 2026-02-04 | v4 / 2026-06-01 | 摘要与作者页；[项目页](https://agile-hoi.github.io/)标注 SIGGRAPH 2026 | 手物交互重建 |
| [SceneSmith](https://arxiv.org/abs/2602.09153) | 2026-02-09 | v2 / 2026-05-30 | 摘要/项目级核查 | 文本到可模拟室内场景 |
| [SAGE](https://arxiv.org/abs/2602.10116) | 2026-02-10 | v2 / 2026-02-20 | 摘要与 [CVPR 正式 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Xia_SAGE_Scalable_Agentic_3D_Scene_Generation_for_Embodied_AI_CVPR_2026_paper.pdf) | 任务驱动场景生成 |
| [MAG-3D](https://arxiv.org/abs/2604.09167) | 2026-04-10 | v1 | 摘要级核查 | grounding / coding 多 Agent QA |
| [E3VS-Bench](https://arxiv.org/abs/2604.17969) | 2026-04-20 | v3 / 2026-06-29 | 摘要/官方代码/项目页；[项目页](https://k0uya.github.io/e3vs-proj/)标注 ECCV 2026 | 3DGS 环境主动视觉搜索 |
| [Skill-3D](https://arxiv.org/abs/2606.07436) | 2026-06-05 | v2 / 2026-06-11 | 方法、训练目标精读 | 场景条件 skills / post-training |
| [SpatialClaw](https://arxiv.org/abs/2606.13673) | 2026-06-11 | v1 | 方法与 action-interface 对照精读 | 持久 Python 状态中的空间推理 |
| [MUSE](https://arxiv.org/abs/2606.14168) | 2026-06-12 | v1 | 摘要级核查 | 约束跟踪与局部场景编辑 |
| [Agentic Real2Sim](https://arxiv.org/abs/2607.19190) | 2026-07-21 | v5 / 2026-10-11 | v1/v5 方法与评测精读 | 实际交互 episode 转仿真 |
| [Agentic Building-Aware Satellite GS](https://arxiv.org/abs/2609.25578) | 2026-09-22 | v1 | 方法、policy selection 与 GT 使用精读 | DSM 目标权重选择 |
| [FACT](https://arxiv.org/abs/2609.37067) | 2026-09-29 | v2 / 2026-10-09 | v2 方法、诊断接口精读 | articulated twin 几何/接触/动力学 |
| [Real2Gym](https://arxiv.org/abs/2609.37089) | 2026-09-29 | v1 | 摘要级核查 | 视频转仿真及可复用技能 |
| [LiteReality-Agent](https://arxiv.org/abs/2610.01863) | 2026-10-01 | v1 | 方法、质量门控、评测精读 | RGB-D 室内数字孪生 |
| [Agentic RSR](https://arxiv.org/abs/2610.10479) | 2026-10-07 | v2 / 2026-10-11 | v1/v2 方法与评测差别精读 | 视频重建、策略执行、sim2real |
| [RoboSimStudio](https://arxiv.org/abs/2610.18835) | 2026-10-13 | v2 / 2026-10-14 | 摘要级核查 | 可编辑 solver/contact/physics toolkit |
| [SceneAgent（Harvard CRG）](https://computationalrobotics.seas.harvard.edu/SceneAgent/) | 本轮未确定 | 项目页现状 | 第一方项目页核查；未确认独立论文日期 | 多来源捕获转可交互仿真 |

没有独立核实录用状态的条目仅作为 arXiv 预印本处理。全部原讨论中的标题均能找到对应第一方记录；但“存在这篇论文”和“这篇已经解决目标问题”是两回事。

## 3. 最接近问题的方法审阅

### 3.1 AREA3D：VLM 不直接掌控任意恢复操作

输入含初始图像及 pose；VGGT confidence 投到体素场，VLM 给遮挡/边界等区域及优先级。固定权重融合后，缓存可见性 mask、贪心候选排序、frustum uncertainty decay 产生视角序列。下游 PGSR 重建；Replica/OmniObject3D 评测。主文算法的更新包含确定衰减，不能概括为每一步 VLM 诊断真实重建失败再选择工具；更没有 view-graph 修复动作。其核心贡献落在 utility representation 与高效 selection，而非工具数量。[v2 §3–4](https://arxiv.org/html/2512.05131v2)

### 3.2 卫星 Agent：实际上是依赖验证集指标的有限配置选择

输入卫星图像、camera models、SAM 建筑 masks。操作只选 baseline / staged / building-specialist 权重计划；改动的是 EOGS 光度损失权重。观测含 mask QA 和候选 DSM 验证结果，输出审计记录。正文明确当前离线选择依赖 candidate-set validation，未主张无需 validation 的策略泛化。DSM MAE 对齐 GT 后计算；四个 JAX 场景。它支持“语义目标改变重建配置”，不能充当已验证的 GT-free sequential recovery。和用户担忧的薄 orchestration layer 很接近。[v1 Operational Setting、Method、Experimental Setup](https://arxiv.org/html/2609.25578v1)

### 3.3 Agentic Real2Sim：此前遗漏的早期直接相关工作

输入有同步视频、标定、robot trajectory、深度与任务。Agent 选对象/keyframe，mask critic 可退回重选，tracking critic 可换初始化；工具执行 SAM、深度、mesh、pose tracking。后续选确定性 object-shift sweep 或看 replay 再修改。输出是可回放仿真 episode，不是 SfM 图。重要限制：DROID100 success 由最多五个候选和三个 VLM judges 的最佳通过结果定义，不是独立几何 GT；不能把它的成功率直接作为相机恢复证据。其优点是明确区分工具执行与有歧义的决策。[v1 §3–4](https://arxiv.org/html/2607.19190v1)

最新 v5 扩至 500 episodes，增加 PSNR/SSIM/ADD-S（只在 349 个可评估结果上计算）、下游机器人用途与三个 one-prompt 对照。judge success 定义仍保留；四模型比较的 100 个 episodes 另有人工核验。作者明确没有独立 actor–critic ablation。one-prompt baseline 没有 AR2S 专用 tools/skills，不能当作严格“同工具预算、仅换控制策略”证明。这些新增结果不能遗漏，也不能追溯为 7 月首版就已具备。[v5 §IV](https://arxiv.org/html/2607.19190v5)

### 3.4 LiteReality-Agent：真实重建的 observe/edit/verify 已经存在

输入 RGB-D scan、相机标定、point cloud、RoomPlan layout。Agent 检索证据并编辑 Room.py，处理材质、fixtures、小物体与资产；检查实际 mesh collision、support、articulation，失败返工。停止主要受调用预算控制，末尾 10% 调用关闭探索。十个场景/620 个采集视角；测渲染、深度、接触与穿插。作者明确按采集视角而非 held-out novel views 测 fidelity。这里相机是已知输入，研究的是结构化室内资产，不是未注册相机恢复；但泛称“Agent 修复 reconstruction”已经重合。[v1 §3–4](https://arxiv.org/html/2610.01863v1)

### 3.5 FACT：不能再把“针对性测量—诊断—修复”当独有机制

从图像生成的 mesh 与 motion videos 建立部件、关节、collision proxy、物理参数。Agent 决定参数化特征、测量区域和方法；数值工具返回 surface/silhouette 等差异；根据诊断改 representation 或重新测量。接触阶段固定 geometry/joints，重分割碰撞代理；动力学阶段固定前三者拟合 response model。重点是局部诊断与可编辑变量的对应关系。其几何参考包含 image-generated mesh，因此不能等同独立真实多视角几何 GT，也不处理 camera graph。[v2 §3](https://arxiv.org/html/2609.37067v2)

### 3.6 Agentic RSR：改场景，冻结相机；深度反馈不是外部 GT

32 视频帧的 VGGT-Ω 输出通过已知 robot URDF 定尺度，四个参考视角用于编辑反馈。Agent 改 Blender geometry/layout/material/light；程序禁止改 robot 和参考相机，以深度、颜色和 coverage 门槛验收，再做 MuJoCo interaction checks。depth reference 来自前面的估计，作者明确不是 independently measured GT。v2 的主重建配置改为 DSH orchestration / Claude Opus 5.5，因此不能把 v1 的单一 coding-agent backend 描述沿用到 v2。不修复相机，但已有外部评价器控制的重建返工。[v2 §3.1](https://arxiv.org/html/2610.10479v2)

### 3.7 SpatialClaw：接口设计本身可以是可检验的研究变量

VLM 每步写 Python cell；persistent kernel 保留数组、depth、mask、pose。后续读取 stdout、变量摘要、渲染图，再写分析。既可调用感知工具，也能用 NumPy/SciPy 组合运算。无需训练；比较 single-pass code、structured calls 与 persistent code interface，在 20 个 QA/空间任务上测效果。最终产物是答案，非改善过的 reconstruction。这不是“多包装几个工具”的唯一例子，但也说明“有 feedback loop”已是常见设计，不能仅靠它主张新颖。[v1 §2–4](https://arxiv.org/html/2606.13673v1)

### 3.8 Skill-3D / Think3D：训练政策与静态几何工具要分开

Skill-3D 从成功轨迹抽取 scene-aware skills，将失败记为 lessons；检索 skills 指导工具使用，再以 SFT + GRPO 训练小模型。reward 含答案、格式、取得规定证据与调用成本；required evidence 由冻结 parser/task type 决定。评价的是空间 QA，未证明提高底层重建质量。[v2 §3](https://arxiv.org/html/2606.07436v2)

Think3D 使用 Pi3 建点云后变换/渲染供 QA。最新 v5 明确每个问题最多调用一次 reconstruction，默认三次 interaction/render，STOP 可提前结束；GRPO 学习探索策略。因此“在固定估计里换观察角度”与“获取真实新观测纠正错误估计”必须分开。v5 不能被误记为 10 月才首发。[v5 §3](https://arxiv.org/html/2601.13029v5)

## 4. 其他分支及它们真正证明了什么

| 分支 | 已核查的机制 | 对当前选题的意义 |
|---|---|---|
| Active3D | hybrid neural-field/Gaussian 状态；分层 uncertainty 融合；EIG view planning、风险感知路径、局部 refinement。[原文](https://arxiv.org/html/2511.20050v1) | 闭环残差与 refinement 不是 LLM 独有；非语言策略必须进入相关工作 |
| Explore until Confident | 语义地图引导探索，多步 conformal prediction 校准停止。[作者页](https://explore-eqa.github.io/) | “什么时候停止”可形成明确统计机制，独立于 toolkit |
| E3VS | 99 个 3DGS 场景、2,014 个 question episodes；控制视角找 QA 证据。[论文](https://arxiv.org/abs/2604.17969) | 重建是评测环境基础设施，非待修复产物；arXiv 摘要为 5-DoF、项目页为 6-DoF，应标版本差异 |
| MAG-3D | planner、grounding/frame-retrieval、coding verification 三类 agent。[论文](https://arxiv.org/abs/2604.09167) | 程序几何验证已有；最终仍是 grounded reasoning |
| SAGE | generators + 语义/视觉/物理 critics，任务驱动迭代生成，验证下游 policy。[论文](https://arxiv.org/abs/2602.10116) | 方法价值可来自可用训练环境与扩展效果；并非从真实 image evidence 恢复相机 |
| SceneSmith | designer/critic/orchestrator 分层布置场景，整合资产生成与物理属性。[论文](https://arxiv.org/abs/2602.09153) | collision/stability 约束闭环已成熟；泛化场景生成与真实重建不可混淆 |
| MUSE | 需求分解、局部编辑、Inspector 检查、三类 memory；AuthorBench 测 preservation。[论文](https://arxiv.org/abs/2606.14168) | “改局部且保留其他部分”的任务协议也是贡献；非 geometry GT recovery |
| AGILE | Agent 选择 keyframes/合成正交视图，再做 object geometry 与 robust pose tracking。[作者页](https://agile-hoi.github.io/) | 直接处理 SfM brittle initialization 的相关邻域；与真实 correspondence 修复不同 |
| Real2Gym | scene/camera 对齐、physics action check、代码技能与失败经验复用。[论文](https://arxiv.org/abs/2609.37089) | 同期 real2sim2real 竞争；本轮仅摘要，不据此判断其模块消融充分性 |
| RoboSimStudio | 显式可编辑 solver/contact/physical parameters，供人和 agent 修正仿真。[论文](https://arxiv.org/abs/2610.18835) | toolkit 可以是系统贡献，但需要用户真正希望做的贡献类型与可验证价值 |
| Harvard SceneAgent | 页面明确支持 COLMAP→Nerfstudio 3DGS 并 inspect/review/correct，包括逐图颜色校准；多来源捕获转仿真。[第一方页面](https://computationalrobotics.seas.harvard.edu/SceneAgent/) | “调用 COLMAP + 检查 + 修正”在系统层已有；公开日期与独立论文尚待核实，不能臆判政策归类 |

## 5. 对“会不会只剩一个 plugin”的研究判断

以下是综合判断，不是文献已证明的新方法，也不代表确定选题。

**工具复用是正常的；问题是去掉工具清单后，是否仍有一个可表述、可证伪的机制。** 有论文贡献不必等于新 backbone，也不必一定训练；但“用更强 VLM 调更多工具”缺少可迁移解释，很容易被通用 coding agent 取代。

可区分四层贡献：

1. **包装层**：把 COLMAP、BA、matcher、segmenter 变成 API。实际价值可以很高，但对当前希望的 3DV 方法论文通常不是充分核心。
2. **工程控制层**：统一状态、缓存、执行、回滚、日志。应该做，但仅增加这层不能证明智能决策解决了新视觉问题。
3. **决策机制层**：例如表示多个几何假设、预测诊断操作的辨别力、在代价与风险下选择新证据、学习跨场景干预价值。需要明确状态、更新、目标及失败条件。
4. **视觉问题层**：例如错误但自洽的匹配如何被可获得的新证据区分；无 GT 的验收何时可靠；局部改动为何能改善全局可观测性。这里的答案应体现 3DV 结构，而非仅是通用 agent prompt。

第 3/4 层不自动新颖：FACT 已有定向测量和 representation-vs-parameter 诊断，Agentic Real2Sim 已有可返回的 perception critics，SpatialClaw 已有可修改分析的执行接口。若后续选题采用这些思想，需要再具体到**相机/对应关系估计中的困难**，而不是换一个目标 domain 就宣称新机制。

一个判断贡献是否薄弱的思想实验：给同样工具、同样图像、同样预算，一个强通用 coding agent 加基础任务说明。如果方法仅靠更多 workflow instructions 获胜，应明确定位为 harness/system；若核心在几何状态表示或推理规则，就应能说明 generic agent 缺少哪一种信息/约束，以及该机制为何对未见场景有效。此处只是判断标准，不是在启动实验。

另一个容易忽略的区别：**内在一致性不是外在正确性。** satellite workflow 能看 GT DSM；RSR 对预测 depth 做一致性；Real2Sim 使用宽松的 best-candidate VLM judging；LiteReality 用采集视角和 scan consistency。它们的反馈各有价值，但不能直接支持“Agent 能识别真正恢复成功”。这可能是比工具调度更值得持续研究的机制问题。

## 6. 尚不能下的结论与检索边界

- 不能以“搜索没有找到完全同名题目”证明 view-graph recovery 是空白。本轮 Agent 文献检索还需与传统 SfM graph selection、robust estimation、self-calibration、active matching、learned optimization 的检索合并。
- 不能因 9–10 月密集出现相关预印本就宣判无法投稿；应分别检查最早公开版本、实际重合的 claim、会议同期政策。
- 不能把 arXiv abstract 当作方法充分性证明。表中明确区分全文方法/评测审阅与摘要级筛选；未复现任何结果，亦未审计所有代码。
- 本轮 broad queries 包括：agentic + reconstruction/recovery/diagnosis；agent/LLM + SfM/structure-from-motion/view graph；原讨论各论文精确标题；近邻论文 related work 追踪。引用仅使用论文、正式会议、作者项目/代码。
- GitHub 可找到固定 COLMAP→3DGS agent orchestration、annotation agent、QA release gate 等工程仓库；它们提示“调用工具”已容易商品化，但不把缺乏正式实验的仓库当成已验证论文结论。

当前最稳妥的研究立场：**暂停“哪个题排名第一”，先判断真实几何恢复中还有哪一种现有方法无法处理的决策不确定性，以及 Agent 能否改变它。** 从文献看，宽泛 agent loop 已拥挤；相机图内部失败恢复仍可能有具体问题可研究，但尚未证明其空白、必要性或最佳技术路线。

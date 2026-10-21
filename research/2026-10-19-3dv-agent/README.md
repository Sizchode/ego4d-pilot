# 3DV + Agent：投稿边界、文献进展与贡献审查

**文献调研截点：2026-10-19；实验状态更新至 2026-10-21。** Baseline-first 阶段已经完成：DA3-Giant、VGGT、π³-X 与 MapAnything 在同一 103-scene manifest 上完成 412/412 次运行，零 execution failure。每个模型的 standard 与 visibility failure mining 均覆盖 103 场景和 10,358 个有向视图对；远端独立审计、本地重算与最终 CSV 逐字节核对均通过。ScanNet++ 的 20-scene nested GT-factor experiment 表明，DA3 pose 最优但 geometry 输给其他模型的八个场景全部 depth-dominant；逐视图 residual 可以定位 depth error，但 raw residual 会系统性漏掉高-support 的复杂视图。当前研究问题已收敛为：support-aware 数值诊断定位异常后，VLM 能否判断局部 failure cause，并在固定预算和 action set 下超过确定性 selector。完整结果见 [final baseline and failure-mining report](final-baseline-failure-mining.md)，精炼定位见 [related-work survey](related-work-survey.md)。

## 阅读导航

- [CVPR 与 ARR/ACL 投稿政策](submission-policy.md)：同期界线、引用义务、实验比较、版本日期与当前投稿节点。
- [SfM / view-graph 文献](sfm-literature.md)：18 项核心工作、5 项补充；传统恢复、学习筛边、神经先验融合。
- [3D Agent 文献](agent3d-literature.md)：20 项工作；主动采集、空间 QA、真实重建、场景编辑与 real-to-sim。
- [诊断与序贯决策](decision-methods.md)：几何可靠性、动态算法配置、实验设计、视觉工具策略学习。
- [验证器与 benchmark 证据](benchmark-evidence.md)：真实退化、OOD、模型自证、独立真值和评测协议。
- [DA3 主实验与 failure mining](da3-main-results-failure-mining.md)：103/103 配置、visibility-controlled diagnostics 与首批失败类型。
- [局部跨模型 failure mining](partial-cross-model-failure-mining.md)：在完整四模型结果到齐前，对共享 HiRoom、ETH3D 与 ScanNet++ 案例作无阈值核对，并记录 GT-extrinsics 因果干预。
- [Snapshot 09 failure review](partial-snapshot-09-visual-review.md)：精确极值案例、ScanNet++ component interventions，以及 test-time diagnostics 对 camera-action headroom 的检验。
- [ScanNet++ camera-action 极值审查](scannetpp-gt-extrinsics-extremal-review.md)：逐 backbone 最大/最小干预收益、归档图像证据，以及 camera-only repair 的反例。
- [Snapshot 10 findings](partial-snapshot-10-findings.md)：完整 7Scenes 的 pose/geometry 决策分歧，以及首批 DTU accuracy–completeness tradeoff。
- [Snapshot 10: 7Scenes visual review](7scenes-snapshot-10-visual-review.md)：28 个完整案例的哈希审计、逐视角 camera-error 一致性、pose/geometry 反例与待验证的 component interventions。
- [Snapshot 11 findings](partial-snapshot-11-findings.md)：DTU 共同人口扩展到 5 scans 后固定模型结论反转，以及集中在少数场景的 oracle headroom。
- [Snapshot 11: DTU five-scan review](dtu-snapshot-11-visual-review.md)：20 个案例的哈希与指标审计、跨模型逐视角 camera-error 排序，以及视觉 sharpness/confidence 无法选择最终 geometry 的反例。
- [Snapshot 11: routing-evidence audit](snapshot-11-routing-evidence.md)：检验 DA3 状态量与全局 RGB 统计能否预测 backbone-oracle F-score gain，并进行 benchmark 内多重检验校正。
- [ScanNet++ 20-scene factor audit](scannetpp-gt-geometry-factors-20.md)：四个 backbone 的 nested GT camera/intrinsics/depth headroom；DA3 的八个 pose/geometry 分歧场景均为 depth-dominant。
- [GRASE 任务边界](grase-comparison.md)：downstream editable-scene repair 与 upstream metric reconstruction recovery 的区别。

下文是交叉综合。专题文件给出逐篇日期、版本、阅读深度与来源；没有把只看过摘要的条目称为全文审计。论文报告的性能没有在本轮复现。

## 1. 投稿政策已经能明确回答

当前公开的 **CVPR 2027 CFP** 将 **2026-09-15 之后上线**的工作一般视为同期工作，并明确投稿不会基于与同期工作的比较而被拒；arXiv 与会议/期刊适用同一规则。作者仍应引用、讨论，并在可行范围内做实验比较；实际影响了投稿的工作仍作为 prior work 承认。边界日期判断由 AC 处理。[官方 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)

**ARR/ACL 的“三个月”同样不是“完全不用管”。** 它免除的是要求新增实验或深入分析的详细比较义务，仍期待可行范围内的引用和讨论。[ACL 官方 Citation 政策](https://www.aclweb.org/adminwiki/index.php/ACL_Policies_for_Review_and_Citation)

CVPR 2026 则是对近期 arXiv 的原则式保护，没有在所核 FAQ 中规定统一三个月窗口。不能直接用 2026 表述替代 2027 的新 CFP。[2026 Reviewer Guidelines](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines)

由此得到的研究判断是：**不必因为每一篇新预印本就无限追加实验或放弃方向，但必须理解直接相关方法。同期保护不能替代原创性论证。** 对 10 月更新的论文，应看首版与相关贡献实际出现的版本；例如 Think3D 首版是 1 月，Agentic Real2Sim 首版是 7 月，不应仅按最新版本日期归为 10 月新工作。[Think3D 记录](https://arxiv.org/abs/2601.13029)、[Real2Sim 记录](https://arxiv.org/abs/2607.19190)

完整的当前投稿节点及无法读取的 2027 作者细则，见政策专题；不把未确认细则写成事实。

## 2. 研究领域的进展，不能只按“Agent 论文”检索

至少有五条线直接决定本选题的边界。

| 研究线 | 已有技术进展 | 对初始设想的约束 |
|---|---|---|
| 错误 SfM 的恢复 | 冲突定位、拆分/合并、撤销早期决策、重新估计 | diagnosis → recovery 并非新任务结构 |
| 图选择与学习鲁棒估计 | 网络流图选择、图网络筛边、多视图消歧、每场景 adaptation | 不能只与原始 COLMAP、固定阈值比较 |
| 神经几何与传统优化融合 | depth/normal priors、局部 feed-forward geometry、虚拟 tracks、augmented BA | “失败时换模型再 BA”已有强技术系统 |
| 可靠性与主动计算 | 置信度修正、概率 uncertainty、refinement gate、扰动诊断、预算控制 | “先测 uncertainty 再选动作”不是充分创新 |
| 3D / 视觉 Agent | 观察—修改—验证、critic 重试、持久代码接口、工具策略训练 | loop、memory、rollback、RL 本身均已有先例 |

这五条线的联系是本报告的综合分类，逐项依据见下面的代表性对照。

### 2.1 必须优先理解的近邻

| 代表工作 | 已经做了什么 | 与“未知相机/匹配图恢复”仍有哪些区别 |
|---|---|---|
| [Recovering Correct Reconstructions…，3DV 2014](https://researchwith.stevens.edu/en/publications/recovering-correct-reconstructions-from-indistinguishable-geometr/) | 定位错误结构、拆分子模型、重新正确合并 | 专注不可区分/重复几何；非跨工具的通用学习决策 |
| [Progressive SfM，ECCV 2018](https://arxiv.org/html/1803.07349v1) | 新输入出现时复查局部结果，复用、拆分、合并、reset/re-estimate | progressive 输入协议；具体算法控制规则已规定 |
| [LFOE，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.pdf) | line-graph Transformer 融合几何及视觉特征筛边 | 已有学到的图处理，但不是任意诊断操作的长期选择 |
| [MP-SfM，CVPR 2025](https://arxiv.org/html/2504.20040v1) | 将单目先验及不确定性纳入几何推断 | 贡献在信息融合；不是上层工具调度 |
| [GLUEMAP，CVPR 2026](https://lpanaf.github.io/cvpr26_gluemap/) | 局部前馈几何、motion averaging、real/virtual tracks 联合优化 | 是强 hybrid reconstruction 系统，削弱简单 fallback 的贡献空间 |
| [Subgraph View Graph Pruning，2026-08-22](https://arxiv.org/html/2608.22054v1) | 运行子图重建产生诊断，再处理不一致连接 | 已有“执行求证再修图”；操作流程与门槛为预定义 |
| [GeoCond，2026-09-16](https://arxiv.org/html/2609.18465v1) | 预测 pose reliability、决定是否 BA，并支持图处理与采集控制 | 已覆盖单步可靠性决策；未完成完整 SfM/SLAM 轨迹验证 |
| [Agentic Real2Sim，首版 2026-07-21](https://arxiv.org/abs/2607.19190) | perception critic 重选帧/初始化，仿真回放反馈修正 | 主要恢复对象/交互 episode；输入与相机标定条件不同 |
| [LiteReality-Agent，2026-10-01](https://arxiv.org/html/2610.01863v1) | RGB-D 场景的观察、编辑、验证与交互结构修复 | 相机和扫描是输入；主要修改可编辑场景资产 |
| [Agentic RSR，首版 2026-10-07](https://arxiv.org/html/2610.10479v2) | 视频到仿真场景，利用视觉/物理反馈反复修改 | 明确冻结参考相机，主要编辑 geometry/layout/material |

9–10 月预印本与更早正式工作分开看：前者需要精确讨论和版本记录，后者已是不能忽略的方法背景。表中的“区别”仅说明任务边界，不代表证明空白。

### 2.2 哪些“新颖点”已不足以单独成立

| 候选表述 | 已有直接覆盖或接近的工作 |
|---|---|
| 根据重建状态补边/删边 | GraphMatch、2018 View-graph Selection、2021 View-graph Construction |
| 用执行反馈修复早期错误 | Progressive SfM、2026 Subgraph Pruning |
| 用视觉语义识别假匹配 | Doppelgangers++、LFOE、MVDG |
| 用 feed-forward model 给失败 SfM 补约束 | MP-SfM、MASt3R-SfM、GLUEMAP、Glob3R |
| 估计可靠性后选择 refinement | LoRA3D、Trust3R、GeoCond |
| 多次推理分歧作为 uncertainty | ReVar3R，且其强调 gauge 与稳定偏差的限制 |
| VLM 调用工具并反思/回滚 | RestoreAgent、AgenticIR、3D/real-to-sim Agents |
| 训练一个会选工具的 policy | TIR-Agent、OPERA、Skill-3D；更早还有动态算法配置 |

逐项原始来源与细节见 [SfM 专题](sfm-literature.md)、[Agent 专题](agent3d-literature.md)、[决策专题](decision-methods.md)。这里不把相邻领域方法误称为已经解决 3D 恢复；它们限制的是通用架构层面的创新表述。

## 3. 对上一轮建议的修正

| 上一轮倾向 | 本次依据后的修正 |
|---|---|
| 恢复重建比主动视角方向明显更空 | 尚无依据支持这个排序；恢复既有旧几何文献，也有新 Agent 近邻 |
| 传统 pipeline 基本正常，Agent 才负责处理失败 | 错误恢复、异常约束处理本来就是 SfM 的研究内容 |
| 多步反馈能形成主要区别 | 反馈很常见；需要指出新的证据、推断机制或决策目标 |
| 最接近的 Agent 只有卫星 workflow | 已发现真实 indoor、robot interaction、articulated twin 等更直接近邻 |
| Oracle 有空间就值得接 LLM | 特权搜索收益不证明实际证据可支持选择，更不证明需要 LLM |
| 用 ETH3D/DTU 官方 split 就有干净任务 | 数据可复用，但 recovery 的输入、失败分布与可见信息仍需界定 |

这些修正不等于否定问题价值。它们意味着现在还不能把一个宽泛 Agent 框架当作已经确定的论文方向。

## 4. Plugin / toolkit 担忧：真正需要区分的两件事

**一个算法可以作为插件交付，并仍有清楚贡献。** 例如 PixSfM 可以接入 COLMAP，但其研究内容是 featuremetric refinement；LFOE 可以表现为筛边插件，但研究对象是图表示与学习过滤。[PixSfM](https://arxiv.org/abs/2108.08291)、[LFOE](https://openaccess.thecvf.com/content/CVPR2025/papers/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.pdf)

真正令人担心的是：去掉工具名和“Agent”一词后，只剩下“模型看日志，按提示词试几个操作”。这种方案的收益可能来自更强 matcher、更长计算、更充分人工规则，难以归因给研究机制。

也不应走到另一极端，认定必须训练新 backbone 才有贡献。新的问题定义、严谨 benchmark、可扩展系统、可验证接口或清晰分析都可能有学术价值；但需要说明其产生的新能力或知识，并和已有工作作实质比较。CVPR 官方也明确希望评估技术贡献与潜在影响，而非只看是否刷新单一精度指标。[CVPR 2026 Reviewer Guidelines — Be Mindful](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines)

可以用下面的思想审查判断当前论点是否清楚，不代表现在开始做实验：

- **去掉模型品牌，贡献还成立吗？** 应能说出学到/估计的量，以及它如何影响几何。
- **给通用 coding agent 同样工具和预算，它缺少什么？** 答案应是特定状态表示、约束、监督或推断机制，而非仅缺少更多 prompt。
- **换一个 reconstruction backend，方法的主张还能成立吗？** 若完全依赖某个工具的几个故障码，可能更适合系统集成定位。
- **成功是怎样知道的？** 训练残差、模型一致性、VLM 评分与独立几何正确性必须区分。

## 5. 本次调研中最重要的技术问题：可验证的几何证据

### 5.1 自洽不等于正确

这不是泛泛的“LLM 会幻觉”问题，而是几何信息结构的问题。例如由同一组绝对位姿定义相对位姿，则

`T_ij = T_i^-1 T_j  ⇒  T_ij T_jk T_ki = I`

无论绝对位姿是否正确，这个闭环都恒成立。它不能作为该组位姿的独立验收。类似地，只在用于拟合的对应点上计算低 residual，可能无法排除错误但自洽的重复结构。

[GeoCond](https://arxiv.org/html/2609.18465v1) 的独立 pairwise estimates 条件、[ReVar3R](https://arxiv.org/html/2610.07883v1) 对稳定系统偏差的限制，以及 [SysCON3D](https://arxiv.org/html/2605.18754v1) 对伪几何支持的研究，都说明“诊断指标客观可算”不等于“足够诊断真错”。

### 5.2 反馈来源决定一个 loop 实际证明了什么

| 反馈来源 | 可以提供什么 | 不能直接推出什么 |
|---|---|---|
| GT geometry / pose | 离线选择与评价准确性 | 真实部署时也能作出同样选择 |
| 预测深度与渲染的差异 | 对当前参考估计的一致性 | 独立真实几何正确性 |
| VLM judge / render critique | 视觉、语义、任务合理性线索 | 精确相机或对应关系正确性 |
| 独立估计/未参与拟合的观测 | 检查当前假设、发现冲突的机会 | 自动排除共同偏差或所有退化 |
| 新拍摄视角/传感器测量 | 可能真正补充缺失观测 | 免费信息；应属于明确的采集协议 |

例子：[Agentic RSR](https://arxiv.org/html/2610.10479v2) 明确说明其反馈深度参考来自前面的重建；[卫星工作](https://arxiv.org/html/2609.25578v1) 依赖候选验证 DSM 指标。[Agentic Real2Sim v5](https://arxiv.org/html/2607.19190v5) 已增加客观指标，不能说它“只有 judge”；但 replay success 的 judge 定义与相机恢复正确率仍是不同概念。

### 5.3 重新计算可以有价值，但必须解释信息从哪里来

对同一批图像重新匹配，可能挖掘之前没有计算的证据；不一定需要新增传感器数据。所谓“独立验证”也不要求所有工具统计上完全独立，而是需要知道它们共享了哪些图像、特征、先验和初始化，避免循环自证。

反之，如果两个几何解释对全部可得观测都等价，增加语言推理或在错误模型里渲染更多视角不能凭空恢复缺失信息。语义先验可以偏向更可能的解释，但这与由观测验证该解释不同。这是本报告的几何推论，不是已经确认的新方法。

## 6. 哪些技术问题可能超出 toolkit，目前证据到哪一步

以下是需要继续收敛的研究问题，不是“已找到的空白”或推荐实施方案。

| 候选问题 | 有机会形成的技术对象 | 最直接的已有覆盖 | 尚未被本次文献解决的判断 |
|---|---|---|---|
| 错误解释存在歧义 | 与相机/对应关系相联系的竞争假设表示及更新 | 重复结构修复、Doppelgangers 系列、概率 uncertainty | 哪一类自然失败在当前证据下确实无法用现有方法区分？ |
| 诊断计算有不同成本/价值 | 面向最终恢复收益的计算/证据选择 | GraphMatch、DAC、Bayesian experimental design、GeoCond | 下一次操作价值能否从可见信息估计？历史何时必要？ |
| 局部修复可能破坏全局 | 包含可观测性、覆盖与误修复风险的决策目标 | view-graph selection、robust SfM、progressive split/merge | 能否给出已有目标未能表达的具体冲突？ |
| 无 GT 时决定保留/回滚/停止 | 可校准、适用条件明确的验收机制 | Trust3R、ReVar3R、SysCON3D、conformal stopping | 能认证的是残差、风险还是实际全局几何？ |
| 视觉语言信息帮助恢复 | 将语义线索转化为可验证几何约束的机制 | LFOE、Doppelgangers++、MVDG、SpatialClaw | VLM 相较非语言视觉/图模型提供了什么额外能力？ |

抽象的“维护 belief、探索信息、考虑预算”也不是新理论。[Deep Adaptive Design](https://proceedings.mlr.press/v139/foster21a.html) 已学习序贯实验设计，[Dynamic Algorithm Configuration](https://arxiv.org/abs/2205.13881) 已学习算法执行中的配置控制。3DV 方法必须说明其观测、坐标规范、图约束、动作副作用或监督结构带来的特殊难点。

**因此不宜现在规定一定要 LLM、一定要 view graph、一定要多 Agent。** 这些应由尚未解决的具体问题决定。若贡献最终是一个可靠的几何判断量，它可以很小；若是系统型贡献，则需要清晰、可复用的任务与证据，而非靠工具数量撑规模。

## 7. 精读顺序：先建立方法边界，再谈新题

为避免大列表失焦，建议围绕下面四组读，不按追新顺序读：

1. **恢复是否已存在**：2014 Recovering Correct Reconstructions、Progressive SfM、2026 Subgraph Pruning。关注错误如何被发现、哪些决策可撤销、何时执行重建。
2. **现代强基线增加了什么信息**：LFOE、MP-SfM、GLUEMAP、VGPA。关注学习表示、先验与不确定性的处理，而非只比较 pipeline 方框。
3. **诊断能否相信**：GeoCond、ReVar3R、SysCON3D。关注实际可见证据、训练标签、GT-free 的精确定义和失效条件。
4. **Agent 的贡献如何与工具分离**：AgenticIR、SpatialClaw、Agentic Real2Sim、LiteReality-Agent。关注固定工具条件下的控制器比较、接口设计和反馈来源。

这四组的逐篇记录已经在专题文件中。本轮调研足以决定：**暂缓选题排序和 toolkit 实现，先把一个具体尚未解决的几何问题写清楚。** 它还不足以替任何候选作“首次”“最值得做”或“无需非语言控制器对照”的承诺。

## 8. 检索方法、范围与剩余盲区

本轮采用原始来源优先的 targeted review，并通过近邻论文参考文献向前追溯。覆盖五条主线中的数十项研究；不是有预注册数据库和 PRISMA 流程的正式 systematic review，也不是专利 novelty search。

| 检索块 | 代表关键词/入口 | 纳入判断 |
|---|---|---|
| 投稿政策 | CVPR 2027/2026 CFP、Author/Reviewer Guidelines；ARR CFP；ACL Citation | 只以官方当届政策解释规则 |
| 几何恢复 | SfM recovery、duplicate structure、progressive reconstruction、view-graph selection/pruning/construction | 改变相机、对应关系、结构或恢复决策 |
| 学习与混合重建 | learned outlier filtering、MASt3R/VGGT、global SfM、monocular prior | 可能消耗原设想 headroom 的强方法 |
| Agent 近邻 | agentic reconstruction、diagnosis/repair、real2sim；原提案精确标题 | 分辨被编辑对象、反馈来源、决策学习 |
| 决策与验证 | uncertainty calibration、refinement gate、active diagnosis、algorithm configuration、restoration agent | 排查通用 agent 结构和诊断量的先例 |
| 评测 | degraded reconstruction、OOD、3D consistency、真实 geometry reference | 区分数据存在与协议适配 |

排除作为技术结论依据的材料：搜索引擎生成概括、聚合站“已解决”判断、未核实的作者/年份、无实验的工程 demo。作者项目页可证明作者自己的发布说明，但不独立证明效果。

主要剩余盲区：本轮没有全面覆盖 SLAM recovery、动态/非刚性 SfM、rolling shutter、自标定、所有主动 SLAM 与 metareasoning 理论，也未审计全部实现。最终若选到这些边界，应做针对性补查。不存在同名论文不是缺乏先例的证明；同名或类似框图也不是贡献完全重复的证明。

## 9. 目前的判断

CVPR 有同期工作保护，我们不需要因为所有新预印本而无限改题；但本次找到的不只是同期工作，还有足以改变定位的早期文献。**“Agent 根据诊断调用标准工具修复重建”目前只是一个系统构想，尚不是足够清楚的研究贡献。**

这条方向仍值得思考的部分，在于几何失败的可辨识性、证据可靠性以及有成本的干预选择。接下来需要收敛的是：哪些真实错误仍无法被现有方法处理，缺少的是信息、表示、优化还是决策，以及拟提出的机制到底改变哪一项。这个判断完成之前，不把工具开发或 oracle 提升当成已经选定方向的理由。

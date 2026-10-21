# SfM、View-Graph 与重建恢复：文献边界审计

检索截止：2026-10-19。目的：判断「Agentic reconstruction recovery / view-graph repair」的实质研究空间，不启动 pilot，不把工具包装预设为贡献。

## 结论与对前文的更正

**“真实重建失败的诊断与恢复尚无人系统研究”不成立。** 2014 年已有直接读取失败 SfM 模型、检测重复结构造成的错误并恢复的工作；2018 年已有随新证据修正早期错误、复用中间重建的在线系统。它们不叫 Agent，但在功能上覆盖原提案的重要部分。[Heinly ECCV 2014](https://www.cs.unc.edu/~jheinly/publications/eccv2014-heinly.pdf)、[Heinly 3DV 2014](https://researchwith.stevens.edu/en/publications/recovering-correct-reconstructions-from-indistinguishable-geometr/)、[Progressive SfM](https://arxiv.org/html/1803.07349v1)

**补边、删边、重估、局部重建、拆分合并、恢复未注册相机、使用神经几何先验，都已有非 Agent 实现。** 因此研究问题必须写出已有方法没有回答的决策困难，而不能把这些操作列为 innovation。下文的“尚未覆盖”只针对所读论文的方法，不代表证明整个领域不存在。

**一个方法可作为 plugin，不等于贡献弱；贡献只有 plugin 接口，才是问题。** MP-SfM 的核心是带不确定性传播的先验与几何约束融合；LFOE 的核心是 line-graph 表示和学习过滤；GLUEMAP 的核心包括虚拟 tracks 与统一优化。它们均能与现有工具整合，算法论点仍独立成立。[MP-SfM](https://arxiv.org/html/2504.20040v1)、[LFOE](https://openaccess.thecvf.com/content/CVPR2025/papers/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.pdf)、[GLUEMAP](https://arxiv.org/html/2605.26103v2)

## 1. 先分开六个不同问题

| 问题 | 未知量与输入 | 典型方法 | 与 recovery 的关系 |
|---|---|---|---|
| Image retrieval / pair proposal | 哪些图像值得尝试匹配 | 外观检索、邻域扩展 | 提议边，不保证几何正确 |
| View-graph construction | 哪些候选可成为可信几何约束 | 几何验证、局部重建生成相对位姿 | 可主动计算新证据，已存在迭代方法 |
| View-graph selection / pruning | 已计算边中保留哪些 | 网络流、循环一致性、学习过滤 | 常是前处理；不同于跨工具诊断策略 |
| Robust estimation | 如何从污染的约束恢复位姿 | robust rotation / translation averaging、BA | 直接吸收或抑制错误，并非等待外部 Agent |
| Post-hoc / progressive correction | 如何撤销已产生的错误结构 | 冲突检测、拆分、重建、合并 | 已有真正恢复工作，最直接的先例 |
| Learned geometric priors | 传统观测不足时引入什么约束 | depth/normal、pointmaps、feed-forward local geometry | 增加信息，不只是改变控制流程 |

这里的分类是本报告的分析。不能因为某论文用 fixed pipeline 就说它“没有反馈”：内层几何优化、循环验证、重新分组、失败重试本身可能是自适应闭环。应明确待提出的 policy 究竟额外决定什么。

## 2. 核心文献地图（18 项）

日期优先使用会议/作者原文，arXiv 日期是首次上传而非会议录出版日。所有效果属于作者报告，未在本次研究中复现。全文核读标注 **M**；仅核原始摘要/作者页面标注 **A**。

| 工作与状态 | 观测 → 操作/方法 → 反馈/学习 | 评价与边界；对本题的意义 |
|---|---|---|
| **Correcting for Duplicate Scene Structure in Sparse 3D Reconstruction**, ECCV 2014；A。[作者 PDF](https://www.cs.unc.edu/~jheinly/publications/eccv2014-heinly.pdf) | 失败重建中的相机、点及 conflicting observations → 后处理识别和纠正重复结构导致的错误。 | 已明确是 reconstruction error detection + correction。此次作者服务器全文下载失败，只将可检索原文摘要中的事实作为结论。 |
| **Recovering Correct Reconstructions from Indistinguishable Geometry**, 3DV 2014；A。[作者机构](https://researchwith.stevens.edu/en/publications/recovering-correct-reconstructions-from-indistinguishable-geometr/) | 场景几何共现 → 定位致错 3D 点 → 拆成子模型 → 正确合并。无需 LLM。 | 直接威胁“先诊断后恢复”的任务 novelty；不等于已解决所有混合失败或预算决策。 |
| **Robust Global Translations with 1DSfM**, ECCV 2014；M（方法部分）。[作者页/论文](https://research.cs.cornell.edu/1dsfm/) | noisy relative translation directions → 多个 1D 投影中的顺序不一致识别 outliers → robust averaging。 | Internet photo collections。它是给定约束的可靠性估计，不是获取新匹配的策略。 |
| **Structure-from-Motion Revisited / COLMAP**, CVPR 2016；M（方法概述）。[论文](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.pdf) | correspondence graph、当前 2D–3D 对应 → 增量注册、三角化、过滤、BA。 | 从定义起就是迭代系统；“多次运行标准几何操作”不能单独区别于它。 |
| **GraphMatch**, 3DV 2017；M。[论文](https://arxiv.org/html/1710.01602v1)、[作者 PDF/venue](https://web.ece.ucsb.edu/~psen/Papers/3DV17_GraphMatch.pdf) | Fisher 相似度与当前图邻域 → sample-and-propagate 新 pair → geometric verification → 更新邻域，继续搜索。 | 评价匹配图构建效率及重建；主动匹配/反馈补边不是新概念。目标主要是发现有效匹配，并非解释失败成因。 |
| **View-graph Selection Framework for SfM**, ECCV 2018；M。[全文](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Rajvi_Shah_View-graph_Selection_Framework_ECCV_2018_paper.pdf) | 图与局部几何/context priors → minimum-cost network flow 选择子图 → 固定 SfM。 | 同时考虑歧义、效率、准确率；cost/预算式图选择已有清晰数学表达，不是“只能靠固定阈值”。 |
| **Progressive Structure from Motion**, ECCV 2018；M。[全文](https://arxiv.org/html/1803.07349v1) | 随输入变化的图、全局/局部旋转不一致 → 重聚类、transfer/split/merge、reset/re-estimate → 继续重建。 | 随机/不利图像顺序及对称场景；已保留撤销早期错误的能力。没有学到的通用跨工具 policy。 |
| **View-graph construction framework for robust and efficient SfM**, Pattern Recognition 114, 2021；M（出版社方法节可见部分）。[原文](https://www.sciencedirect.com/science/article/pii/S003132032030515X) | 已有 matches → verified maximum spanning tree → 局部重建/注册 → 新相对几何/边 → 迭代扩图。 | 一般与歧义图像集；不是只基于 retrieval 的静态图。局部执行结果驱动图更新已经存在。 |
| **TC-SfM: Robust Track-Community-Based SfM**, arXiv 首发 2022-06-13；M；本次未独立确认正式 venue。[全文](https://arxiv.org/html/2206.05866v1) | track-community/context → 找歧义 segment → pose-consistency 验证 → 分块重建和双向一致性合并。 | 歧义/Internet/人体场景。作者明确弱连接部分可能被隔离、丢失相机，参数也需场景类型适配。 |
| **Global SfM Revisited / GLOMAP**, ECCV 2024，arXiv 2024-07-29；M。[全文](https://arxiv.org/html/2407.20219v1) | relative rotations 和 feature tracks → rotation averaging → 相机与点联合 global positioning → BA。 | 不应把它简单描述成传统两步 translation averaging。作者列出对称导致旋转错误、错匹配/匹配失败等剩余失败。 |
| **MASt3R-SfM**, 3DV 2025，arXiv 2024-09-27；A。[作者机构页](https://europe.naverlabs.com/research/publications/mast3r-sfm-a-fully-integrated-solution-for-unconstrained-structure-from-motion/)、[arXiv](https://arxiv.org/abs/2409.19152) | foundation model 的局部 3D/matches → 低内存全局对齐，并复用模型做检索。 | 学习先验加全局求解已是完整路线；不能把调用 MASt3R 初始化当新的机制。 |
| **Doppelgangers++**, CVPR 2025，arXiv 2024-12-08；M。[全文](https://arxiv.org/html/2412.05826v2) | MASt3R 3D-aware features → Transformer pair classifier → 集成标准/MASt3R SfM。 | 更丰富的日常场景训练、VisymScenes、geotag-based 验证；“用图像高层线索识别错误匹配”已有强前例。 |
| **Learning to Filter Outlier Edges in Global SfM (LFOE)**, CVPR 2025；M。[全文](https://openaccess.thecvf.com/content/CVPR2025/papers/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.pdf)、[代码](https://github.com/DmblnNicole/LFOE-GlobalSfM) | 预估全局旋转、relative geometry、DINOv2 embeddings → line-graph Transformer edge classifier → translation filtering。 | 1DSfM/PhotoTourism/ScanNet；跨数据评估。几何+图像上下文不是 VLM 特有优势；还含 oracle filter。 |
| **MP-SfM**, CVPR 2025，arXiv 2025-04-28；M。[全文](https://arxiv.org/html/2504.20040v1)、[代码](https://github.com/cvg/mpsfm) | depth/normal 先验及不确定性、图像对应 → lifting、depth consistency、扩展 BA 与先验细化。 | 针对低重叠/低视差/对称；恢复来自新的可观测约束与正确融合，不是多试几次相同求解器。 |
| **GLUEMAP: Global SfM Meets Feedforward Reconstruction**, CVPR 2026，arXiv 2026-05-25；M。[全文](https://arxiv.org/html/2605.26103v2)、[作者页](https://lpanaf.github.io/cvpr26_gluemap/)、[代码](https://github.com/colmap/gluemap) | retrieval+消歧 → local feed-forward stars → motion averaging → real/virtual tracks augmented BA。 | 多种挑战及大规模图像；强 hybrid baseline。图连通性相关的动态阈值也已在方法中。 |
| **Robust Global SfM via View Graph Pruning**, arXiv 2026-08-22；M；未确认录用。[全文](https://arxiv.org/html/2608.22054v1) | recursively partition → 子图 GLOMAP → loop/pose consistency → 跨子图 PnP/RANSAC 剪边 → 最终 GLOMAP。 | 歧义、序列、无序及渲染评估；明确是执行局部重建获得诊断证据，不只是静态看 graph。 |
| **VGPA: Learning Global Camera Poses from Noisy View-Graphs**, ECCV 2026；arXiv 2026-09-08；M。[全文](https://arxiv.org/html/2609.09491v1)、[作者页](https://vgpa-sfm.github.io/) | edge-conditioned GNN 聚合 noisy poses → 全局 cameras → test-time consistency adaptation → triangulation/BA，可 reintegrate 被删相机。 | 不依赖 GT 的学习与跨数据评估；非语言模型能处理图污染和相机覆盖。不是“只有一个前向 pass”，完整 inference 含每场景优化。 |
| **MVDG**, arXiv 2026-10-01；M；未确认录用。[全文](https://arxiv.org/html/2610.01098v1) | VGGT 多视图特征 → covisibility classifier → confidence-based early exit → SfM。 | 多视图消歧及计算自适应；没有跨外部工具的长时域选择，但已覆盖多视图 context/置信度调算力的宽泛 claim。 |

## 3. 最接近工作的方法审计

### 3.1 Progressive SfM：最容易被忽略的旧前例

方法第 2.2–2.6 节值得直接阅读。全局图不永久丢弃当下可疑约束，局部图在每轮 robustify；拓扑变化触发局部重建，群组转移复用已有结构；局部结构和图推断的旋转冲突会触发重置。**保留可逆性、积累证据后重新判定、执行反馈纠错，都已经写进算法。** 区别只能放在新的状态表征、策略目标或适用范围，而不能只说“传统流程是一次性的”。[原文](https://arxiv.org/html/1803.07349v1)

### 3.2 LFOE：学习修图、视觉特征与 oracle 都已有

全文第 4–5 节：原图边转为 line graph 节点，再做二分类；重叠聚类用于控制图规模，推理时投票。一场景训练采用 COLMAP 重建产生标签。其 oracle 按 GT 相对平移角误差删边；作者特别说明，去除退化配置还能超过此 oracle，所以它并非整个 SfM 的严格上界。**这支持前文“oracle 要谨慎定义”的判断，但说明该思路也不是本提案独有。** [原文](https://openaccess.thecvf.com/content/CVPR2025/papers/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.pdf)

### 3.3 2026 Subgraph Pruning：诊断本身已经调用重建

第 3 节先递归拆分并实际运行 GLOMAP，用循环及 relative-to-absolute consistency 判断是否继续；跨块再用 PnP/尺度对齐提供边验证依据。**“先跑局部求解，再利用结果修图”已有直接覆盖。** 它仍是预定算子顺序、预定阈值的程序；论文没有研究从异质诊断操作中学习选择下一操作的长期价值。后一句是对方法范围的判断，不是全领域空白结论。[原文](https://arxiv.org/html/2608.22054v1)

### 3.4 MP-SfM：有些失败是约束不足，不是控制器不聪明

它用先验提供缺失约束，并把先验误差传播到几何优化。这个贡献形态说明：一个说得清的信息缺口及其求解方法，比罗列 segmentation/matching/BA 工具更重要。代码明确先验 uncertainty 需要 calibration；不能把任意模型 confidence 都当成已校准的成功概率。[原文](https://arxiv.org/html/2504.20040v1)、[作者代码说明](https://github.com/cvg/mpsfm)

### 3.5 GLUEMAP：强 hybrid 系统已经吞掉很多 fallback headroom

使用局部星形图而非全图 attention，并将真实和虚拟 tracks 融入优化，解决先验局部有效但全局不稳的矛盾。作者明确仍依赖 feed-forward 局部质量、当前模型不能处理鱼眼图像，并指出严重外观变化/临时物体上的评估仍待扩充。**不能把“COLMAP 失败时调用一个 feed-forward model，再 BA”作为充分贡献。** [方法及局限](https://arxiv.org/html/2605.26103v2)

### 3.6 MVDG / VGPA：最新进展对宽泛 claim 的约束

MVDG 的 early-exit confidence 预测的是是否同意最终深层输出，不自动等于几何 GT 正确率；其廉价验证代理的相关性，也不是每场景可依赖的正确性证书。[MVDG 第 4、5 节](https://arxiv.org/html/2610.01098v1)

VGPA 的无 GT consistency training、target-scene adaptation、被过滤相机 reintegration 表明“学习+test-time adaptation+相机恢复”已有组合。它也承认 MegaDepth/1DSfM 的 reference 主要来自 COLMAP，独立 GT 在小规模数据补充。[VGPA 第 3、4 节](https://arxiv.org/html/2609.09491v1)

## 4. 哪些弱点有一手证据，哪些还只是研究假设

| 议题 | 已有证据 | 可以提出但尚未证明的研究问题 |
|---|---|---|
| 错误但自洽的结构 | 重复结构修复系列；2026 pruning 明确讨论 false consensus | 能否通过有成本的干预区分多种几何解释？ |
| 剪边损害 coverage | TC-SfM 作者明确 weak connectivity 导致丢视角；LFOE 评价相机 loss | 能否同时考虑证据可靠性与维持可恢复性，而非局部二分类？ |
| 场景参数泛化 | ETH CVG 的复现项目明确同一参数不能覆盖全部歧义场景 | 学习的状态/策略是否比逐场景调阈值更稳定？ |
| 先验不可靠 | MP-SfM 要求 uncertainty calibration；GLUEMAP 依赖局部预测 | 如何判定哪类证据值得相信、值得追加计算？ |
| 闭环操作选择 | GraphMatch、Progressive SfM、局部子图方法已有闭环 | 异质诊断操作的次序是否存在不可被静态评分替代的价值？ |
| 语言模型必要性 | LFOE、Doppelgangers++、VGPA 已用非语言学习融合复杂信息 | 什么任务信息必须受益于预训练语言/视觉语言能力，而非图网络或搜索？ |

复现项目的一手证据是[作者自己的实验与结论](https://github.com/cvg/sfm-disambiguation-colmap)，并非它对所有后来方法的断言。其结果说明老方法的固定参数限制，不能直接推出 LLM 已经是答案。

## 5. 对“toolkit-only contribution”担忧的具体判断

以下是本报告的研究判断，而非文献已证实的新方法：

1. **弱版本**：标准工具集合 + LLM 读日志 + prompt 中规定触发规则。控制逻辑、几何信息及恢复能力都来自工具；LLM 的不可替代作用难以成立。
2. **仍偏弱**：学一个操作分类器，但动作标签来自哪个工具在 GT 上最好，没有解释状态歧义与序列依赖。这很可能是 algorithm selection，而非需要 Agent 的问题。
3. **可能有独立贡献的版本**：显式表示多个几何解释及其不确定性，学习某个诊断计算对区分解释的价值，并将证据获取成本、误修复风险和最终几何质量共同纳入决策。其难点是 observation aliasing、信用分配和可观测性，不是 API 数量。
4. **另一种有效贡献**：一个新几何可靠性量/校准方法、可验证的失败状态表示或新的诊断协议，使已有工具能更正确地使用。这可能成为优秀 3DV 方法，是否叫 Agent 不必先决定。

第 3 项仍需与 active perception、Bayesian experimental design、metareasoning、algorithm configuration 的文献对齐；本报告不把“没在这 18 篇看到”写成 novelty。应由总研究继续核查。

## 6. 本题相关的新近补充与覆盖边界

- **G-MASt3R-SfM**, arXiv 2026-06-22，原作者页面注明 accepted to ICIP 2026：graph-based pruning + local-to-global multi-stage optimization，ETH3D。[原文](https://arxiv.org/html/2606.22856v1)、[状态](https://arxiv.org/abs/2606.22856)
- **Glob3R**, arXiv 2026-07-10：冻结 Pi3X 加 dense matching head、跨 window tracks、global averaging/BA。进一步显示 feed-forward 与 classical geometry 融合非常活跃；此次只核摘要与状态，不作全文细节评价。[原始记录](https://arxiv.org/abs/2607.09225)
- **VGGSfM**, CVPR 2024：端到端可微的 tracks、camera estimation、BA，是与“工具拼装”不同的学习路线；需作为 broader reconstruction 背景。[作者论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_VGGSfM_Visual_Geometry_Grounded_Deep_Structure_From_Motion_CVPR_2024_paper.pdf)
- **Doppelgangers**, ICCV 2023：Doppelgangers++ 的直接前身，已定义视觉相似但不同 3D surface 的二分类及数据。[CVF](https://openaccess.thecvf.com/content/ICCV2023/html/Cai_Doppelgangers_Learning_to_Disambiguate_Images_of_Similar_Structures_ICCV_2023_paper.html)
- **PixSfM**, ICCV 2021：可作为 COLMAP add-on 的 featuremetric refinement。提醒我们“插件形态”和“有无算法贡献”是两个维度。[作者 arXiv](https://arxiv.org/abs/2108.08291)

本子报告没有全面覆盖动态 SfM、rolling shutter、自标定、SLAM loop-closure recovery、可微求解器、所有 rotation averaging 理论。没有找到异质 SfM 工具的 LLM 长时域诊断控制器，不构成不存在的证明。搜索还出现一个期刊日期标为 2026-11-15 的 triangulation-sensitivity view selection 条目，因相对截止日期的公开时间未核清，未纳入证据主表。

## 7. 目前可以成立的定位

“SfM 会失败，需要恢复”是已知问题；“图像语义 + 几何诊断”是已有信息组合；“反馈后改图/重建”是已有算法结构；“神经先验 + 优化”是主流进展。

剩余值得继续文献核查的候选，是**在多个可能失败解释之间，如何选择会产生区分性证据的计算，并保证这种选择改善最终几何而非仅改善自洽指标**。这可以成为决策/学习研究，但当前不能断言一定需要 LLM，也不能断言已经是新任务。

下一步应先完成跨领域近邻核对并给出可证伪的机制定义；本报告不建议现在开始大规模 toolkit 或 pilot 实现。

# 3DV + Agent：诊断、可靠性与序贯决策的方法边界

检索截止：2026-10-19。本文只处理“控制器是否有研究贡献，以及几何反馈能否支持控制”的方法问题；不启动 pilot、不指定最终选题。证据来自论文全文、官方会议论文页、作者代码。下文区分“文献事实”“推论”“待验证假设”。新近 arXiv 的作者报告不等于已经独立复现。

## 1. 当前证据支持什么结论

**研究判断：把 COLMAP/VGGT/分割/匹配封装成工具，再给 LLM 一段提示词，不能自动形成强方法贡献。更值得审查的技术对象是：系统在不知道几何是否正确时，如何取得独立证据、维护多个解释，并为最终几何质量选择有成本的干预。**

这也不是尚无人涉足的抽象思想。Dynamic Algorithm Configuration 已经研究根据执行状态动态配置算法；Bayesian experimental design 已经研究根据过去观察选择下一次实验；dual control 已经把动作的改善作用和探测作用统一。近期 3D 文献又已经覆盖可靠性估计、置信度再标定和选择性 refinement。因此，后续不能用“多步”“自适应”“有反馈”“用 uncertainty”四个词充当 novelty。[DAC](https://arxiv.org/abs/2205.13881)、[DAD](https://proceedings.mlr.press/v139/foster21a.html)、[dual control](https://epubs.siam.org/doi/10.1137/16M1069936)、[GeoCond](https://arxiv.org/html/2609.18465v1)。

## 2. 最相关原始文献：具体做了什么，哪里与设想重叠

### 2.1 LoRA3D：已有“利用几何反馈修正模型自身”的具体方法

**事实。** LoRA3D 初稿为 2024-12-10；ICLR 2025 论文。它从少量未标定 RGB 图像的预测出发，做鲁棒多视图点图对齐、重估置信度、筛选伪标签，再进行 LoRA 场景适配。第 4.2 节将置信度转为可优化权重，以残差导出闭式重加权，而不是让模型语言判断“好不好”。方法明确利用不同视图对的共同观测；低重叠中的过度自信是其动机之一。[全文](https://arxiv.org/html/2412.07746v1)、[ICLR 正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/6db7c49b14da8006892fda7350d76b6a-Paper-Conference.pdf)。

**与本题关系。** “重建→验证→修正→再重建”早已有不依赖 LLM 的算法内容。它的亮点不是工具数量，而是几何目标与置信度更新之间的关系。仅把这个循环换成 agent orchestration 不足以证明新技术。阅读深度：全文方法 3–4 节、实验协议及原始正式论文检索核对。

### 2.2 Trust3R：可靠性 head 与选择性修正已有明确先例

**事实。** 初稿 2026-05-19，arXiv 作者元数据和官方作者仓库均标为 ICML 2026。冻结 MASt3R 主干后增加 Normal–Inverse–Wishart evidential head，得到多元 Student-t 点分布；另有 gated residual head 修正点图。训练使用真实 3D 点，评估包括 risk–coverage、sparsification 和几何精度。第 3.3 节明确：建模单点 XYZ 内部协方差，但不显式建模跨像素协方差。[全文](https://arxiv.org/html/2605.19539v1)、[作者元数据](https://arxiv.org/abs/2605.19539)、[代码](https://github.com/phai-lab/Trust3R)。

**与本题关系。** “让几何模型知道哪里不可信”本身已有竞争；像素级可靠性也不自动等于全局 pose graph 的失败概率。可借鉴其概率输出和评价方式，但不能把 evidential 参数直接当成真实世界误差的保证。阅读深度：全文 3 节及评估协议。

### 2.3 GeoCond：与“读几何诊断、决定是否 BA”高度直接重叠

**事实。** 2026-09-16 arXiv 预印本。基于预测的视差、重叠、基线、旋转、camera-head 修正及 native confidence，用小 MLP 输出 pose reliability 和 refinement gate。uncertainty 可用 permutation variance、GT pose error 或独立 pairwise graph 的 cycle residual 监督；**BA gate 的 Eq.13 则以训练时真实 pose error 判断 BA 是否改善。** 推理不需要 GT。应用包含 BA gating、pruning、rotation averaging 权重和采集停止。作者承认完整下游 SfM/SLAM 轨迹评价仍是后续工作。[全文 3–5 节](https://arxiv.org/html/2609.18465v1)。

**与本题关系。** “用诊断决定何时调用优化器”已不能作为独占创新。研究若推进到多步修复，应解释为什么这个单步 gate 不够，以及历史反馈究竟增加了什么信息。本文不把其 GT-free uncertainty distillation 扩大解释为整个训练流程 GT-free。阅读深度：全文方法、关键消融、limitations。

### 2.4 ReVar3R：反复运行的分歧也可能是假 uncertainty

**事实。** 2026-10-06 arXiv 预印本。作者推导 pointmap 在不同相似变换坐标系中的方差含与真实错误无关的项，因而先将多次预测做鲁棒 Sim(3) 配准，再计算分歧。核心可不使用标签，但完整融合/标定版本使用 held-out 标签。论文清楚区分 label-free core 与完整版本，也明确报告：不能识别稳定的系统偏差，跨域数值标定不能直接迁移。默认评估还使用 per-view frame stabilization，应与全局几何评价区分。[全文 3–7 节](https://arxiv.org/html/2610.07883v1)。

**与本题关系。** 扰动一致性是候选证据，不是真值；多模型或多次运行“都同意”也不能证明正确。若主张新的诊断机制，必须解释它能发现哪类不一致、对什么错误失明。阅读深度：方法、supervision 分层、负结果和评价坐标约定。

### 2.5 VGGT uncertainty audit：native confidence 必须实测含义

**事实。** 2026-06-15 arXiv 预印本《Uncertainty Quality of VGGT: An Analysis on the DTU Benchmark Dataset》检查 raw feed-forward 输出的置信度、过滤和误差排序。它明确没有对 BA 后输出使用原始 uncertainty 做同样解释，因为优化中没有相应传播。文中还指出论文符号和实现里 confidence/uncertainty 的方向不能混用。[全文](https://arxiv.org/html/2606.16479v1)。

**审查意见。** 不直接照搬该文对损失函数“是否概率化”的概括。更稳妥的结论是：训练得到的 loss weight 不等于已经验证过的 pose failure probability，校准、排序和实际决策应分开评价。阅读深度：3–5 节。

### 2.6 DUSt3R 官方实现：confidence 是训练变量，不是用户定义的成功率

**事实。** 官方 `ConfLoss` 的实际运算是 `pixel_loss * confidence - alpha * log(confidence)`，另有点图归一化。代码里的注释不完全可靠，应以运算和论文共同解释。[作者代码](https://github.com/naver/dust3r/blob/main/dust3r/losses.py)。

**推论。** 在理想单项最小化且不考虑输出约束时，最优权重与残差成反比；这仍不推出它是“相机正确的概率”，也不能跨任务直接作停止阈值。真正的区分是被学习的量、评价目标、部署决策之间是否对齐。

### 2.7 DAC：动态工具/参数选择早已有成熟问题定义

**事实。** 《Automated Dynamic Algorithm Configuration》，JAIR 75，2022，研究在算法执行过程中按状态和实例改变参数，系统讨论学习策略、强化学习和算法控制。配套案例覆盖 CMA-ES 步长、规划启发式选择和训练学习率；DACBench 为这类问题提供统一环境。[论文](https://arxiv.org/abs/2205.13881)、[作者实验仓库](https://github.com/automl/2022_JAIR_DAC_experiments)、[官方 DAC 文档](https://automl.github.io/DACBench/main/source/dac.html)。

**与本题关系。** 若观测就是低维统计、动作只是从几种 matcher/BA 配置中选择，首先属于 algorithm control，不需要语言模型才能成立。新增 LLM 必须用其泛化、语义证据或工具适配能力解释增量，不能只与固定默认参数比较。阅读深度：原始摘要、定义文档、作者实验说明；未逐页精读 67 页全文。

### 2.8 DAD：学习多步信息采集策略，不依赖 LLM

**事实。** Deep Adaptive Design，ICML 2021。用历史 design–observation 对预测下一次实验设计，以全序列 expected information gain 的 contrastive bound 训练策略，避免每轮在线求后验；可学非贪心策略。原版假设显式 likelihood、给定隐参数后实验条件独立，且设计变量连续。[正式全文，尤其 3–4 节及 limitations](https://proceedings.mlr.press/v139/foster21a/foster21a.pdf)。

**与本题关系。** “为了后续修复而先做诊断”可以联系此类问题，但实际重建工具常离散、有副作用、耗时不一，likelihood 未知，不能把 DAD 原版直接视为现成答案。将“真实几何恢复收益”与信息采集关联，比泛泛最大化 entropy reduction 更贴近问题。阅读深度：全文方法及限制。

### 2.9 Dual control：动作同时改变状态和提供信息

**事实。** La 等，《Dual Control and Online Optimal Experimental Design》，SIAM J. Scientific Computing，2017，研究控制性能与估计信息的共同目标，并讨论何时额外的信息采集成本值得付出。[出版社原文页](https://epubs.siam.org/doi/10.1137/16M1069936)。

**与本题关系。** 重新匹配既可能改善图，也揭示原假设是否可信；这类“双重效果”是经典问题，而非 agent 时代才出现。3D 工作需要提供几何约束、观测模型或学习目标上的特殊处理。阅读深度：原始摘要和问题描述，未全文复现推导。

### 2.10 HDRAgent：视觉重建工具调度 + 反馈循环已经存在

**事实。** 2026-06-08 arXiv 预印本；针对多曝光 HDR。MLLM 分析运动、曝光、遮挡，匹配历史案例/工具知识，动态路由 alignment 与 reconstruction 工具，再把质量评价写入反馈记忆。训练集记忆可使用 full-reference 指标；测试 loop 明确剔除 GT，仅使用 no-reference 指标和 MLLM 感知诊断。还有生成式局部对齐。[全文 III 节](https://arxiv.org/html/2606.09110v1)。

**与本题关系。** 不属于 SfM，却直接限制“重建工具箱 + 诊断 + memory + retry”作为通用结构贡献。生成式视觉对齐也不能直接当作真实 3D correspondence 证据。阅读深度：III 节模块定义、train/test 反馈边界。

### 2.11 PtychoBench：自然语言领域知识不等于可靠视觉诊断

**事实。** 《Adapting General-Purpose Foundation Models for X-ray Ptychography in Low-Data Regimes》，2025-11-04 预印本，NeurIPS 2025 AI4MAT workshop。提供 391 个专家标注样本，分视觉 artifact detection 与文本 parameter recommendation，比较 SFT/ICL；两类任务的最佳适配策略不同。[全文](https://arxiv.org/html/2511.02503v1)、[作者代码](https://github.com/crumeike/PtychoBench)。

**与本题关系。** 可以支持“诊断和建议应分别验证”的方法论动机；不能直接证明通用 3D 场景中的 VLM 一定失败，也不能把专家问答成绩等同于执行后重建收益。阅读深度：任务与适配设置、作者结果说明。

### 2.12 Agent Lightning：训练工具控制器已有通用基础设施

**事实。** 初稿 2025-08-05，将 agent execution 组织成 MDP，用分层 RL 和 credit assignment 将复杂轨迹用于训练，并分离训练和执行。[原始论文](https://arxiv.org/abs/2508.03680)。2026 年已有 v1.0 harnessed agentic RL 版本。[2026-08-18 论文](https://arxiv.org/abs/2608.17528)。

**与本题关系。** “我们用 RL 训练一个会调工具的 LLM”不能单独承担方法创新；几何观测、训练目标、可验证的信用分配或泛化机制才需明确。该文不证明 3D 控制成功。阅读深度：原始论文摘要及作者仓库说明，未全文审计系统。

### 2.13 更直接的视觉工具策略先例：RestoreAgent → AgenticIR → TIR-Agent / OPERA

这组论文比泛型 agent 框架更适合检验“贡献是否只是包装工具”的问题。

**RestoreAgent，NeurIPS 2024。** 初稿 2024-07-25，已经用 MLLM 判断退化种类/程度、选择 restoration task、安排顺序、选择模型并执行。[正式论文页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c78f639424b8d89ceb4f2efbb4dfe4f4-Abstract.html)。阅读深度：正式摘要及初稿元数据。

**AgenticIR，ICLR 2025。** 初稿 2024-10-23。五阶段为 perception、scheduling、execution、reflection、rescheduling；有针对质量判断微调的 VLM、探索后总结的领域经验、工具执行失败后的 rollback。全文将其解释为在处理顺序上的 depth-first tree search。[全文 3 节](https://arxiv.org/html/2410.17809v1)、[正式论文](https://openreview.net/pdf?id=3RLxccFPHz)。阅读深度：方法与能力获取设计。**因此“有反馈、有回滚、有记忆、会重新规划”都不是 2026 年尚新的通用架构论点。**

**TIR-Agent，2026-03-29 预印本，2026-07-08 v2。** 将工具调用策略用 SFT 后接 RL 训练；采用探索用的轨迹扰动、多指标自适应 reward，并处理工具调用吞吐。[原始论文页](https://arxiv.org/abs/2603.27742)。阅读深度：原始摘要和版本信息。**“把调工具的 policy 真正训练起来”也已经是直接的视觉研究方向。**

**OPERA，2026-05-21 预印本。** 用 RL 直接优化组合计划，以最终 restoration quality 为 reward，同时让执行工具随 agent 选择联合训练，处理独立工具在序列组合下的不协调。[原始论文页](https://arxiv.org/abs/2605.22104)。阅读深度：原始摘要。**planner–tool joint training 也不能仅靠换成 3D 工具就声称通用创新。**

这组工作不是 SfM 方法，不能据此说 3D 恢复已经解决；它们意味着后续必须把创新落在 3D 的证据、约束、可辨识性或训练问题上，而不是重新画同一种 agent 框图。

## 3. 不应混为一谈的四类研究对象

以下是本次综合推论，而不是上述论文已经替我们验证的结论。

| 研究对象 | 实际学习/推断的问题 | 已有方法最直接的竞争 | Agent 可能新增的内容 |
|---|---|---|---|
| 可靠性估计 | 当前预测可能错在哪里、错多大 | Trust3R、GeoCond、ReVar3R、几何 residual | 对未知失败提供独立证据，处理共同偏差 |
| 一次性配置/动作选择 | 看初始状态选一次处理 | algorithm selection、GeoCond gate | 工具集合变化时的适配或多模态判断 |
| 序贯算法控制 | 用执行反馈调整下一步动作 | DAC、RL、模型预测控制 | 变化的工具/场景中可复用的策略表示 |
| 主动诊断与干预 | 先取得什么证据，才能决定怎样修复 | BOED、POMDP、dual control | 3D 特有的可辨识性、证据依赖结构、几何与语义融合 |

“工具接口稳定”与“策略是 hardcode”是两件事。固定动作空间可以让研究可比较；依据手写角落规则安排动作则属于 baseline 或实现选择，不能伪装成学习贡献。反过来，给每个操作发一个自然语言名字，也不自动使控制问题需要 LLM。

## 4. 关键的几何陷阱：一致性不是正确性

**代数事实。** 若相对位姿全由同一组绝对位姿导出，例如 `T_ij = T_i^-1 T_j`，则 `T_ij T_jk T_ki = I` 对任意绝对位姿都成立。这个 loop closure 不能验证那些绝对位姿是否真实。需要独立估计的边、未参与拟合的对应关系或其他外部证据，才有机会构成诊断。GeoCond 的 independent pairwise estimates 条件因此是实质约束，不是实现小节。

**统计推论。** 同一 backbone 的多次预测可能共享偏差；不同工具如果共享特征、训练数据、初始化，也不独立。不能把“重复验证次数”当成独立证据数量。低 residual、高 confidence、良好 cycle consistency 可以同时出现在错误但自洽的模型中。

**几何推论。** 已注册比例上升不一定说明相机变准；删掉困难点可以降低误差同时损失覆盖；把每个 component 单独对齐可能掩盖全局关系失败。一个在线 evaluator 必须明确自己验证的是局部拟合、全局结构、覆盖，还是绝对正确性。

**信息边界。** 对真正不可辨识的输入，仅增加推理 token 不会产生缺失的观测。控制器可能需要选择额外约束、请求新证据，或者承认无法决定；但“何时不可辨识”本身也不能靠语言自信度替代几何分析。

## 5. 一个帮助筛选论文的形式化视角

以下是分析框架，不是推荐立即实现的方案。

设 `x_t` 是可见的 pipeline 状态，`z` 是真实几何与失败解释，`h_t` 是此前操作和观察，`a_t` 是有成本的诊断或恢复动作。可考虑 `b_t(z)=p(z | h_t)`，或者学习其近似表示。重建误差是隐藏的目标，日志和一致性是有噪声的观测。

研究问题不是简单最大化“下一步 residual 下降”，而是：在剩余预算下，某动作带来的新证据会怎样改变后续选择，以及这种改变能否降低最终几何损失。纯信息增益不一定有用——某个不影响最终相机精度的 latent variable 很不确定，也不值得花钱测。

这里要避免两种概念包装：

1. 软件环境的内部状态通常可完全读取，不应仅因“用了 Agent”就宣布 POMDP。需要明确隐藏量到底是什么、历史为何必要、状态压缩损失了什么信息。
2. 若一个小数值模型就能从当前图统计预测最优动作，复杂 belief state、语言 memory 或多 agent 都不是先验必需。必须由错误结构和观测依赖来决定表示。

## 6. 有可能成为方法贡献的候选，及其举证负担

这些是**待验证研究假设**，不是已经确认的文献空白，也不是方向排序。

| 候选技术对象 | 应明确的新东西 | 需要排除的替代解释 |
|---|---|---|
| 可干预的失败状态表示 | 表示错误的多个竞争解释，以及操作后的证据更新，而非只列出失败标签 | 现有 uncertainty head + 当前统计量已足够 |
| 任务相关的证据选择 | 预测某次额外匹配/独立估计对最终恢复决策的价值 | 只是统一增加计算或选择最大 uncertainty |
| 带几何约束的序贯策略 | 利用 gauge、图结构、相关观测与动作副作用，学出跨场景可迁移决策 | 普通 DAC/RL/短搜索就可达到同样效果 |
| GT-free 在线验证与拒绝机制 | 能指出什么错误可检测、什么错误无法认证，并有校准适用条件 | 用训练误差、自己拟合的 residual 或 GT 做自证 |
| VLM 的独特证据贡献 | 图像/语义帮助区分数值诊断无法区分的几何解释，并影响可执行选择 | 提升来自额外输入、更多计算或更强底层 matcher |

选择其中任何一个都需回到最直接的几何文献，不宜同时堆五个模块。一个 plugin 可以承载好的算法，但 plugin 的工程形态本身不是学术贡献；一个很小的 head 也可以承载清楚的技术问题，模块数量与贡献强弱无直接关系。

## 7. 审计 GT、oracle 与在线决策时的边界

| 阶段 | 可允许的信息 | 不能混入的方法结论 |
|---|---|---|
| 离线训练 | 清楚声明的训练集 GT、动作执行结果、专家标注 | 把使用 GT 的训练包装为完全自监督 |
| 校准/模型选择 | 与测试场景分离的校准标签 | 在 test scene 选择阈值后再称零样本泛化 |
| 在线运行 | 当前输入、工具执行、合法历史与约定可用的新观测 | 用 GT 决定回滚、停止、最佳分支或失败原因 |
| 最终评价 | 独立真实 pose/geometry、完整计算成本 | 把在线 proxy 的改善直接叫作真实恢复 |
| Oracle 分析 | 特权 GT；必须单独命名并标明搜索范围 | 把一步贪心 oracle 当多步最优上界 |

一步动作的最大 GT 增益也不能衡量纯诊断动作的价值，因为诊断可能只改变未来决策。反之，存在特权 oracle 收益也不能证明可见证据能选出该动作。两者之间的缺口，正是需要研究而不是默认已解决的部分。

## 8. 阅读完成后仍然开放的判断

本次证据已经足以撤回“几何诊断 + 标准恢复操作 + Agent 闭环的位置明显空”这一强说法，但不能据此断言整条方向没有机会。**最重要的未决问题是：现有可靠性/优化方法遗漏的失败，是否存在可获得而尚未被合理利用的证据，以及序贯干预能否利用这些证据做出不同决策。**

目前不应先承诺“必须 LLM”或“必须 view graph”。如果最终贡献是新的可验证几何诊断、信息采集目标或学习机制，Agent 是合理执行方式；如果贡献只能描述为增加几个工具、写更长 prompt、让多个 agent 分工，则用户对 toolkit 化的担心成立。

本文件不是全领域的排他性 novelty 搜索证明；传统 robust SfM/view-graph selection 与 agentic 3D 相邻工作需结合其他专题报告共同判断。这里对 GeoCond、ReVar3R 等新近工作采用精确版本日期，避免把作者新预印本误记成已长期确立的结论。

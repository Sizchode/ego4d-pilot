# CVPR 与 ARR/ACL：同期工作、引用与比较义务

核查日期：2026-10-19。这里只解释公开投稿/评审规则，不把会议政策当作研究创新性的证明。目标届次尚未由用户指定；以当前已公开的 CVPR 2027 CFP 为主，2026 规则仅用于说明差异。

## 1. 直接结论

**CVPR 2027 有明确的同期工作规则，但不是滚动的“三个月豁免”。** 官方 CFP 使用固定界线：2026-09-15 之后上线的论文一般视作同期工作；此规则同时适用于 arXiv 和会议/期刊论文。仍应引用、讨论，并在可行范围内进行实验比较；实际影响了投稿的工作仍须作为 prior work 承认。高度相似稿件会进一步核查，发布日期等边界由 AC 判断。[CVPR 2027 CFP，Contemporaneous Work](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)

**ARR/ACL 的三个月规则也不是“完全不用管”。** 它减免的是需要追加实验或深入分析的详细比较义务，而非删除相关工作、忽略实际影响来源的许可。引用与可行范围内的讨论仍然需要。[ACL 官方 Citation 政策](https://www.aclweb.org/adminwiki/index.php/ACL_Policies_for_Review_and_Citation)

## 2. 逐项对照

| 问题 | CVPR 2027 | ARR/ACL |
|---|---|---|
| 如何定义同期 | 固定日期：2026-09-15 **之后**上线，一般适用 | 投稿截止前三个月内出现 |
| arXiv 与正式论文是否区别适用 | 同期规则不区别两者 | 同期比较豁免也覆盖两者；引用优先正式发表版本 |
| 是否可完全不引用 | 不可据此推导；应引用、讨论 | 不可据此推导；应在可行范围内引用、讨论 |
| 是否必须补齐所有实验 | 在可行范围内比较；明确不会基于与同期工作的比较而拒稿 | 不强制追加实验或深入分析式的详细比较 |
| 真正影响了本人方法的工作 | 必须承认为 prior work | 必须给予 credit |
| 是否保证录用或自动获得创新性 | 否 | 否 |

来源：[CVPR 2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)、[ARR CFP — Citation and Comparison](https://aclrollingreview.org/cfp)、[ACL Citation 政策](https://www.aclweb.org/adminwiki/index.php/ACL_Policies_for_Review_and_Citation)。

## 3. CVPR 2026 的旧表述不能直接套用到 2027

2026 reviewer FAQ 对近期 arXiv 工作采用原则式保护：不要求充分讨论/比较近期报告；漏引 arXiv、没有超越其性能、或 arXiv 已有类似想法，不应单独构成拒稿理由。实际启发本人的工作仍应承认。该 FAQ 没有给出统一的三个月界线。[CVPR 2026 Reviewer Guidelines](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines)

2026 Author Guidelines 的 dual-submission 段落与 arXiv FAQ 需结合阅读，不能抽取一句“arXiv 不算 prior art”来解释所有届次、所有情境，更不能把它扩张成可以借用别人的方法而不引用。[CVPR 2026 Author Guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines)

## 4. 投稿流程：哪些已核实，哪些尚未核实

CVPR 2027 是向会议 OpenReview 直接投稿；不是 ARR 的“先滚动评审、再 commitment”两阶段机制。当前公开节点如下：

| 节点 | CVPR 2027 当前官方日期，AoE |
|---|---|
| Paper enrollment / registration | 2026-11-10 |
| 正文投稿 | 2026-11-16 |
| 补充材料 | 2026-11-23 |
| Reviews released | 2027-01-25 |
| Rebuttal | 2027-01-25 至 2027-02-01 |
| Final decisions | 2027-02-25 |

来源：[2027 官方 Dates](https://cvpr.thecvf.com/Conferences/2027/Dates)、[2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)。ARR 的 review/commitment 分离见 [ARR CFP](https://aclrollingreview.org/cfp)。

2027 CFP 还要求作者具备更新后的 OpenReview profile，并规定禁止实质相似稿件的并行投稿。页面的 dual-submission 段落目前把 review period 结束写为 2027-02-22，而同页及 Dates 把最终结果写为 2027-02-25；这里保留这一差异，不自行修正官方文本。[2027 CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)

本轮访问 `/Conferences/2027/AuthorGuidelines` 返回 404，ReviewerGuidelines 也未成功读取。因此没有把 2026 的页数、rebuttal 新实验限制等细节冒充为已确认的 2027 完整规则。2027 CFP 的同期政策本身已可读取，结论不依赖缺失页面。

## 5. 对这次选题调研的实际意义（分析，不是官方规则）

1. **把评审保护与选题价值分开。** 某篇 9 月末/10 月论文可能受同期政策覆盖，但其方法已经公开；我们仍应弄清楚它做了什么，避免把它当作不存在。
2. **不能因任何一篇新预印本就立即否决方向。** 需要比较任务输入、目标、机制、训练、反馈及验证范围，而非只比较名称中有没有 agent/reconstruction。
3. **旧工作仍是决定性参照。** 2018 年 progressive recovery、2025 年 learned graph filtering 等不因后来添加 Agent 而消失。
4. **记录首次公开时间及所读版本。** 较早预印本的新版本不能机械地当成新工作重新计算界线；会议日期、arXiv 首次提交、项目页首次公开可能不同，边界最终由 AC 判断。
5. **不作无限追新承诺。** 本报告的知识截点是 2026-10-19；未来投稿前再按目标届次核对公开政策与真正相关的新工作。

## 6. 三种义务的区别

- **引用/credit**：来源是否相关，是否实际影响了方法。
- **比较/实验**：在相应会议政策与可获得代码、数据、计算条件下，合理需要做哪些比较。
- **研究贡献**：去掉工具封装和“agent”标签后，论文带来了什么新的知识、算法或可靠的评估问题。

前两项由政策部分约束；第三项必须通过研究内容论证，不能靠时间窗口替代。

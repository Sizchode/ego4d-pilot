# Tool selection 与 VLM 必要性：精简证据审计

日期：2026-10-19。依据本目录已经核查的原始文献，不追加大范围检索、不运行实验。本文的“已有方法覆盖”不等于该类问题已经完全解决；“候选”不等于已经证明需要 VLM。

## 最需要纠正的五条 claim

| 过度 claim | 证据与应采用的表述 | 对工具/评测的影响 |
|---|---|---|
| **1. 几何方法已处理好 F1/F3/F6/F7，剩余 F2/F4/F5 真需要 VLM。** | 这只是工作分组，不能作能力结论。低重叠、退化和短 tracks 仍困难；反过来，重复结构、动态、镜面已有视觉/几何 specialist。正确表述是“前一组先用成熟专门方法作默认工具；后一组检查是否还有上下文归因收益”。[PixSfM](https://arxiv.org/abs/2108.08291)、[DG++](https://arxiv.org/html/2412.05826v2)、[RoMo](https://openaccess.thecvf.com/content/ICCV2025/html/Goli_RoMo_Robust_Motion_Segmentation_Improves_Structure_from_Motion_ICCV_2025_paper.html) | 不写按错误类别 hardcode 的 VLM routing；先看同工具、同证据下实际差异。无 VLM 数据结果前只能称候选。 |
| **2. VLM 比现有方法多看了图像/语义，因此能诊断重复结构。** | DG++ 使用 MASt3R 视觉几何特征，MVDG 使用多视图上下文，LFOE 使用 DINOv2+图结构。OCR+文本空间布局也已用于 loop verification。[MVDG](https://arxiv.org/html/2610.01098v1)、[LFOE](https://openaccess.thecvf.com/content/CVPR2025/html/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.html)、[TXTLCD](https://arxiv.org/html/2410.15869v1) | 基线必须拿到相同图像、ROI、geometry、可用帧。不能只比较纯数字阈值。工具若内部用这些模型，应独立报告其收益。 |
| **3. VLM 认出运动物体、镜子或柜门，再调用 mask/solver，就是未被做过的修复贡献。** | Dynapo 已 VLM→视频 masks→几何优化；FunREC 已 VLM 分阶段/关节类型→动态重建；3D Visual Illusion 已 RGB+几何预测→VLM reliability/fusion。[Dynapo](https://arxiv.org/html/2512.05398v1)、[FunREC](https://arxiv.org/html/2604.05621v2)、[Illusion depth](https://wu-yuwei-bit.github.io/paper/neurips2025_yao.pdf) | 这些可作为已有工具/基线，不能把封装调用重复计为我们的模型贡献。以新词命名 workflow 也不改变此结论。 |
| **4. 对象重复、镜后虚拟房间或反常深度说明相机错了；mask 后 BA 就是修复。** | 对象 track 碎裂可发生于正确 pose；固定平面镜能产生自洽虚拟几何；静态海报纹理可以被经典多视图正确恢复。必须分开对应身份、物理表面解释、相机估计三个层次。[ThinkGraphs](https://arxiv.org/html/2606.31471v1)、[Reflect3r](https://arxiv.org/html/2509.20607v2) | 工具应能报告干预了哪些 observations/tracks/cameras；比较“只改对象/深度”和“重估相机”。禁止以内部残差降低或点数减少替代正确恢复。 |
| **5. 标准 benchmark 分数或正确解释就足以证明诊断/Agent 的价值。** | IMC/1DSfM 是 SfM 派生参考；DG/Visym 主要 pair/geotag 评价；known-pose 对象图不是未知相机恢复。各集也没有完整 cause→action GT。解释流畅不是因果归因正确。[IMC 官方](https://www.cs.ubc.ca/research/image-matching-challenge/2020/data/)、[1DSfM](https://research.cs.cornell.edu/1dsfm/)、[DG++](https://arxiv.org/html/2412.05826v2) | 先写明原生协议、参考来源与输入权限；固定 backend/工具预算，衡量几何+注册覆盖，再单独验证诊断。公开的 test GT 也不能进入 Agent 或接受/回滚逻辑。 |

## 两个最值得保留的 VLM 候选

### A. 实例关联错误与相机错误互相混淆

**具体问题：** 两个同款包装盒被纹理 tracks 错误关联，或同一把椅子因为轨迹/关联问题出现两份。当前几何可能不可靠，因此不能仅按 3D overlap 合并。要判定哪种身份解释成立，并定位污染相机的具体对应关系。

**实证边界：** ThinkGraphs 确实展示 VLM 用图像与几何区分椅子/电器碎片和相似叠放盒子；也展示被 AABB 误导的错误。它输入已知 pose RGB-D，修对象图，**没有证明相机恢复中的 VLM 优势**。[原文 §3.3/App.C](https://arxiv.org/html/2606.31471v1)

**可能增量（假设）：** 在现有 pose 错误时，把可见实例证据转成区域/track 的可验证关联限制，并选择改身份关联还是重估相机。语义同实例不等于像素同点；必须经 matcher/几何验证落实。

**强非语言基线：** [IRIS-SLAM](https://arxiv.org/html/2602.18709v1) 的 joint geo-instance association、同 ROI 的 re-ID/跨视图实例特征+图模型、DG++/MVDG、LFOE。另加 ThinkGraphs-style 一次 VLM critic，区分 VLM 感知增益与多步控制增益。

**工具边界：** 读取带对应/track ID 的原图与子图；查询指定实例跨视图证据；执行指定区域 rematching、track split、受影响相机重估。工具不能自行决定正确身份后再让 VLM“选择”答案。共同使用的 specialist 输出作为输入保留，不能只给 VLM 更好的 masks。

**benchmark 位置：** DG/Visym 用于表面身份误关联；室内对象问题可借 ScanNet/ScanNet++ 注释，但必须明确额外未知相机协议。二者不是同一个原生测试。若 re-ID+geometry 相当，VLM 主张失败；若一次 VLM 相当，多步 Agent 主张失败。

### B. 局部真实状态变化与全局注册错误互相混淆

**具体问题：** 柜门前后处于不同状态，门叶跨时段静态 tracks 把两段相机拉错；或者柜门其实没动，是相机注册错误造成门重影。两者症状接近，修复不同。VLM 的候选作用在部件关系、时序状态与空间冲突的联合归因，不是认出“门可以转”。

**实证边界：** FunREC 已用 VLM 进行阶段划分和关节类型预测，Dynapo 已 VLM 动态 mask。当前没有证据证明反馈驱动 VLM 比专用运动模型更好地解决“状态变化 vs. 错误相机”归因。[FunREC](https://arxiv.org/html/2604.05621v2)、[Dynapo](https://arxiv.org/html/2512.05398v1)

**可能增量（假设）：** 在缺少完整运动过程、初始 pose 不可靠时，用门叶/门框/稳定墙面之间的证据推断应该解除哪些静态约束；保留稳定背景，重估 pose，再检验局部关节解释。若所有证据都缺失，应保留不确定性而非常识补全。

**强非语言基线：** [RoMo](https://openaccess.thecvf.com/content/ICCV2025/html/Goli_RoMo_Robust_Motion_Segmentation_Improves_Structure_from_Motion_ICCV_2025_paper.html)+固定 SfM、[SM4RT](https://arxiv.org/html/2607.22534v1)、[Flow4R](https://arxiv.org/html/2602.14021v1)、同 masks 的多刚体/关节拟合与 joint optimization。VLM 近邻必须包括 FunREC/Dynapo 和一次性诊断。

**工具边界：** 查询部件/背景分组的 tracks 与分段残差；执行受约束的运动模型拟合、解除指定跨状态 track、固定后端相机重估。让数值求解器估轴和角度，VLM 不直接写入猜测 pose。

**benchmark 位置：** 3RScan 跨次扫描身份/变化可以支持场景变化，参考不是全部独立测量；TUM/Bonn 支持动态相机恢复但不原生覆盖完整柜门关节 GT。FunREC 数据/代码可用性须另核实。不能把所有动态序列都归到这个更窄机制。

## 实际封装时如何避免 toolkit 与 Agent 混为一谈

仅建议接口原则，不是新增实现：所有工具返回受影响的 image/region/edge/track IDs、执行结果、有效支持和计算成本；诊断标签是 hypothesis，不当 oracle。相同工具分别接固定流程、非语言控制器、一次 VLM、多步 VLM。评价先看恢复正确性与覆盖，再看运行成本；把“工具本身更强”和“选择/修正诊断更好”分开。

优先默认调用成熟几何工具处理 feature refinement、匹配扩展、初始化/BA、MVS/融合问题；这不是宣称其已解决，也不是排除 VLM。只有 A/B 在同工具比较中出现可重复增益，才值得把对应诊断机制称为 VLM contribution。镜面/屏幕保留为负对照与扩展讨论，不宜同时铺成第三个核心贡献。

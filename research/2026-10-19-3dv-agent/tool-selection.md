# 从 failure taxonomy 筛选工具与 VLM 的职责

日期：2026-10-19。状态：基于已核查文献的工具范围与评测设计；尚未实现工具、运行实验或提交排行榜。

## 1. 筛选结论

不能把“现有方法仍会失败”直接解释为“必须使用VLM”。本轮文献支持三种处理方式：

1. **几何/专门视觉算法优先，直接复用。** F1可靠对应不足、F3观测几何不足、F6相机估计、F7深度融合。它们并非已解决，但没有证据支持把数值求解交给VLM。
2. **需要图像内容，但已有强非语言方法。** F2重复结构、F4一般动态、F5一般反光/透明。专门方法本来就在看图像，不能把它们描述成只看残差。
3. **值得验证VLM增量的窄子问题。** 错误几何状态下的实例身份冲突；运动历史/语义解释与当前残差发生矛盾。尚无证据证明它们只能由VLM解决。

## 2. 全部类别的实现取舍

| 类别 | 已有处理路径 | VLM判断 | 实现决定 |
|---|---|---|---|
| F1 可靠对应不足 | 学习匹配、dense matching、多尺度、图像质量/曝光处理 | 识别模糊/低光通常不需要通用VLM；无纹理造成信息缺失也不能靠解释补成真值 | 复用匹配与质量工具，保留原始观测用于验证 |
| F2 错误身份/假共视 | Doppelgangers++、MVDG、LFOE、instance re-ID/几何约束 | 最值得研究的是多个证据冲突时定位具体错误关联，而非输出笼统same-place | 专门消歧作为默认候选生成；VLM audit作为可替换模块 |
| F3 低视差/低重叠/缺桥接 | 图连通性、三角化几何、检索和重新配对 | VLM不能创造未观察的共视。不同对象的错误连接属于F2，不用单一标签处理 | 实现有预算的rematching；只访问协议允许的图像池 |
| F4 动态/状态变化 | RoMo、DynaSLAM、ReFusion；语义VLM近邻Dynapo | 普通人/车mask已很拥挤。较有价值的是短暂停止/相机跟随前景、静态复杂表面被误删的矛盾诊断 | 先复用时序motion support；VLM时序audit第二优先 |
| F5 非朗伯/反射/错觉 | 材质检测、专用stereo/反射模型、VLM depth fusion | 认出镜子不是定位相机错误；已有3D Visual Illusion覆盖VLM可靠性融合 | 暂不作为第一版主功能；选定depth或pose目标后再适配 |
| F6 相机/全局估计 | 正确内参模型、鲁棒估计、BA、global SfM、hybrid SfM | 几何优化有直接目标；VLM可选择检查项，但其必要性没有证据 | 复用固定求解后端并保留状态快照 |
| F7 密集深度/融合 | ACMM、MVSFormer++、MVSAnywhere、visibility consistency | “这是墙”不是可评估的精确深度。工具收益与VLM收益需分开 | 复用局部重估与融合；camera-repair主线的第一版可暂不启用 |

来源：[GLUEMAP](https://github.com/colmap/gluemap)、[Doppelgangers++](https://arxiv.org/html/2412.05826v2)、[MVDG](https://arxiv.org/html/2610.01098v1)、[LFOE](https://openaccess.thecvf.com/content/CVPR2025/papers/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.pdf)、[RoMo](https://arxiv.org/abs/2411.18650)、[Dynapo](https://arxiv.org/html/2512.05398v1)、[ACMM](https://openaccess.thecvf.com/content_CVPR_2019/html/Xu_Multi-Scale_Geometric_Consistency_Guided_Multi-View_Stereo_CVPR_2019_paper.html)、[MVSAnywhere](https://openaccess.thecvf.com/content/CVPR2025/papers/Izquierdo_MVSAnywhere_Zero-Shot_Multi-View_Stereo_CVPR_2025_paper.pdf)、[3D Visual Illusion](https://wu-yuwei-bit.github.io/paper/neurips2025_yao.pdf)。

## 3. 第一优先的VLM工具：audit_visual_identity

### 要解决的具体错误

同一建筑的不同立面或两个同款物体通过重复纹理被错误关联，当前相机解释可能内部自洽。VLM检查关联的原始图像证据，判断是否出现物理身份矛盾，例如不兼容的固定雕塑/标识、不同周边布局、同时出现的独立实体。完全对称、没有额外信息的场景应输出证据不足。

输入应包含：完整原图与标记crops；稳定region/observation/track/edge IDs；关联的原始匹配；几何支持和冲突；可访问的其他视图。只给渲染截图会隐藏真实观测并放大当前几何的偏差。

输出应包含：支持same-surface/different-surface/insufficient-evidence的假设，引用具体图像区域，待验证的关联IDs，尚缺证据，建议验证操作。不能让VLM根据文字描述任意生成精确keypoint坐标；region到observations的映射由分割、跟踪与几何模块落实。对象相同不等于所有特征是同一3D点。

执行由标准模块完成：重新估计候选匹配；拆分具体错误tracks/移除错误edges；重估受影响相机。VLM判断只是候选诊断，不直接认定恢复成功。

### 与已有工作差在哪里

[ThinkGraphs](https://arxiv.org/html/2606.31471v1) 已用VLM修对象track，但相机已知；[Breaking Déjà Vu](https://arxiv.org/html/2607.12818v1) 已用VLM审计VPR候选；[IRIS-SLAM](https://arxiv.org/html/2602.18709v1) 已融合instance与几何。可能研究增量是错误相机状态下的观测级定位与端到端相机修复，尚未验证。

### 评测

先在DG/VisymScenes检验错误pair及SfM消歧，用PhotoTourism的明确版本协议检查一般场景代价。需同时报告正确注册覆盖与错误合并；仅提高pair分类分数不等同SfM修复。DG/Visym缺乏独立精密dense GT，不承诺一个数据集证明所有相机/表面结论。GT labels与geotags由离线evaluator保管。

必须有DG++/MVDG、图模型或re-ID、相同输入的一次性VLM、完整GLUEMAP，以及同一工具与总预算下的非语言选择器。借用现成模型产生诊断收益属于工具能力，不能全部归为controller贡献。

## 4. 第二优先：audit_motion_context

具体问题是“低残差前景真的静态吗”和“高残差墙面真在运动吗”。[WildGS-SLAM failure cases](https://arxiv.org/html/2504.03886v1) 已报告后者的Shopping墙面及前者的Wandering人物；但这些案例来自无GT轨迹的iPhone定性集，不是TUM/Bonn现成失败标签。

输入为原始时间序列、对象motion tracks、前景/背景各自拟合的相机支持、历史可见性与残差。输出分段状态假设和应检查的观测；短时静止的人可能是当前有效的几何支持，不能仅按person类别删除。静态地图目标与动态对象重建目标必须明确。

优先在TUM/Bonn验证，使用相同RGB或RGB-D条件比较RoMo+backend、DynaSLAM/ReFusion、适配的Dynapo。Bonn的独立静态扫描补充几何指标。该工具只在身份主线之外确有数据支持时纳入，避免两条任务一起膨胀。

## 5. 最小可执行工具范围

这是功能接口设计，不是已实现API或占位代码。复用真实算法，适配真实输入输出；不写按dataset/scene名称分支的修复规则。

| 能力 | 输入范围 | 实际输出 | 优先级 |
|---|---|---|---|
| inspect_reconstruction | 当前图、相机、tracks、原图；无GT | 可追溯diagnostic packet、候选问题对象 | 必需基础设施 |
| rematch_and_verify | 明确image pairs/regions与已有图像池、算法配置 | 经过几何验证的候选matches及证据 | 第一版 |
| edit_associations | 经定位的track/edge修改提案 | 新关联状态、受影响相机集合与变更记录 | 第一版 |
| solve_cameras | 子图或全图、内参假设、固定求解器配置 | 实际pose/sparse geometry、失败状态、成本 | 第一版 |
| audit_visual_identity | 原始图像、关联ID、几何状态 | 有证据引用的身份假设或证据不足 | 第一版VLM候选 |
| estimate_motion_support / audit_motion_context | 原始时间序列、对象tracks、相机状态 | motion support与可选VLM状态假设 | 动态分支第二阶段 |
| refine_surface | 固定相机、指定区域与source views | 重估深度/融合表面 | 几何榜单扩展阶段 |

状态快照、重新评估和回滚作为共享执行设施。几何验证使用独立观测支持、coverage等可得量；这些是代理证据，并不保证真值正确。GT pose、激光几何和榜单指标只用于离线评估。

## 6. 如何解释榜单收益

至少分开四个结果：原始强后端；相同工具的固定恢复；相同工具的一次性VLM诊断；相同工具和总预算的多步controller。若只前两者有收益，就报告工具/系统收益。若VLM只改善诊断但不改善pose/geometry，不应宣称重建恢复有效。

调参与停止条件在开发集确定；测试时不使用GT选动作。dataset/scene名称不作为policy特征，不手工为榜单场景指定修复。原生calibrated MVS、unposed SfM、RGB-D动态轨迹是不同赛道，分别执行协议与报告，不合成一个虚构总榜单。第一版筛选建议以F2身份冲突为研究变量，保留F1/F3/F6的真实几何工具作为共同执行能力。

# Benchmark-first：SfM 与重复结构的 failure taxonomy

核查日期：2026-10-19。原始论文/项目页为事实来源；“症状→诊断→动作”是本调研的任务分析，不是原论文已经给出的失败频率统计。**困难条件、输出错误和因果诊断必须分开**。例如重复纹理是条件，错误合并是输出错误，跨实例 tracks 污染是可能原因。

## 五个常用数据集/协议

| 数据与原生任务 | 参考来源及限制 | 论文明确的挑战/实例 | 适合证明什么，不适合证明什么 |
|---|---|---|---|
| **ETH3D high-resolution multi-view**：原生为校准 MVS；后续论文将训练场景用于三角化、定位及 SfM pose 评测 | 3D 几何为 Faro 激光扫描；图像先由 COLMAP 初始化，之后对固定扫描作图像对齐并优化内外参。**几何独立测量，但相机参考经估计/对齐得到，不是全部独立 mocap**。部分反光/透明及动态区域被清理或 mask。 | 原论文强调自由 6DoF、视角变化、室内外、细枝/电线等细节；PixSfM supplementary 以 **Courtyard** 展示短 tracks 的三角化误差、平面噪声。后者明确是特征定位噪声放大，不是语义误识别。 | 很适合几何/位姿精度；不能把 calibrated MVS 结果当未知相机恢复。不能声称专门涵盖大量动态、镜面失败。 |
| **IMC PhotoTourism (IMC-PT)**：wide-baseline pair pose 与小集合多视图 SfM | 全图像池 COLMAP 重建得到 **SfM pseudo-GT**，再用小子集评价；不是独立测量。原 2019–2021 challenge test 当时隐藏；官方 2022 年已发布 full test 参考，不能仍称该旧版本全部隐藏。 | 不同相机、视角、光照/天气、遮挡；原论文指出 dominant planes，比较退化处理；低共视使匹配更困难。**Sacre Coeur / St Peter’s Square** 有不同共视难度的明确图例。 | 可测小视图下的 registration/pose 成功率；对已经被参考 SfM 排除的极难图像缺乏覆盖保证。不是重复结构修复专项。 |
| **1DSfM**：给 noisy epipolar graph 求全局相机平移/完整 SfM；图像另提供 | 作者明确真实集合没有已知 GT，比较 **Bundler-based sequential SfM reference**；geotags 只把参考置于近似米制坐标。发布图数据已限制在单 connected component。 | **Gendarmenmarkt** 是明确失败：中心建筑两侧相似穹顶，跨建筑 wide baseline，跨组边少，平均得到坏模型；原文另指出 self-consistent outliers 不在方法能力内。 | 适合图稀疏/异常边/translation averaging；固定发布图本身不评估初始 retrieval 漏边。不能把接近 Bundler 称独立真实精度。 |
| **Doppelgangers (DG)**：同一/不同物理表面的二分类 | Wikimedia 朝向/类别标注 + COLMAP putative matches 挖掘；图像级类别信息派生 pair labels，**不是逐像素/完整相机 GT**。翻转负样本仅用于训练。 | **Big Ben** 四面被匹配后压成一个面；**Arc de Triomphe** 正反面雕塑不同，顶部重复元素产生错误匹配。也含不同地点的建筑复制品。 | 适合错误 image pair 与证据定位；不能仅凭分类 AP 宣称最终 SfM 修复。真实照片的方向标签仍可能存在共视边界歧义。 |
| **VisymScenes / DG++ reconstruction evaluation**：日常场景 pair 消歧与 SfM 集成 | pair labels 由 GPS/IMU、视向和内参几何规则挖掘；最终以 Mapillary geotag probes 或场景自身 geotags 对齐的 inlier ratio 检验。**独立元数据有价值，但有噪声，不是高精度 dense/pose GT**。 | 从地标转向住宅、办公楼、街道；**Belvedere (Vienna)** 中旧 DG 输出失败，DG++ 得到两个 clean components；**Church on Spilled Blood** 展示阈值过强造成分裂/注册减少的权衡。 | 最贴近真实视觉歧义的 SfM 验证；geo inlier ratio 适合在同一场景、相同 probes 下比较，不宜跨场景直接排名。分成更多 component 有时是正确消歧，不能一律视为错误。 |

### 原始来源

1. ETH3D：[原论文 §3–4](https://www.eth3d.net/data/schoeps2017cvpr.pdf)、[官方数据格式](https://eth3d.ethz.ch/documentation)、[官方构建流程](https://github.com/ETH3D/dataset-pipeline/blob/master/README.md)。注意另有 ETH3D **SLAM** benchmark，其 mocap 轨迹协议不同：[SLAM overview](https://eth3d.ethz.ch/slam_overview)。
2. ETH3D sparse 使用：[PixSfM 正文](https://openaccess.thecvf.com/content/ICCV2021/papers/Lindenberger_Pixel-Perfect_Structure-From-Motion_With_Featuremetric_Refinement_ICCV_2021_paper.pdf)、[supplement App.A](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Lindenberger_Pixel-Perfect_Structure-From-Motion_With_ICCV_2021_supplemental.pdf)。其三角化实验给定相机，而 localization 给 sparse map；[Detector-Free SfM supplementary](https://openaccess.thecvf.com/content/CVPR2024/supplemental/He_Detector-Free_Structure_from_CVPR_2024_supplemental.pdf) 另有 ETH3D multi-view pose AUC。必须写明采用哪个协议。
3. IMC：[原论文 §3–4](https://arxiv.org/html/2003.01587v3)、[原 challenge 数据说明](https://www.cs.ubc.ca/research/image-matching-challenge/2020/data/)、[维护者更新的数据页](https://www.cs.ubc.ca/~kmyi/imw2020/data.html)。更新页记录若干因数据不一致被移除的场景，应锁定版本。
4. 1DSfM：[作者项目页](https://research.cs.cornell.edu/1dsfm/)、[作者 PDF §7 Limitations / Fig.5](https://research.cs.cornell.edu/1dsfm/docs/1DSfM_ECCV14.pdf)。PDF 经下载后 pdftotext 核查；web 浏览器直接读取失败。
5. DG：[原论文 §3–4 / Fig.1–3](https://arxiv.org/html/2309.02420v1)、[作者项目页](https://doppelgangers-3d.github.io/)。
6. VisymScenes：[Doppelgangers++ §3.1、3.3、4 / Fig.5–6](https://arxiv.org/html/2412.05826v2)。

## 从这些 benchmark 归纳的错误类别

下表**不是已经统计过的频率排序**。症状只能提出假设，不能单独确定原因；同一低注册率可能来自漏匹配、模糊、视角不重叠或错误初始化。

| 输出错误 | 能覆盖它的数据/文献 | Agent 可以观察的症状 | 需验证的原因及候选动作（分析建议） | 有竞争力的对照 |
|---|---|---|---|---|
| **注册不足/图断裂** | IMC 小集合；1DSfM 弱跨组约束；DG++ 过度 pruning | 未注册图、弱 cuts、匹配覆盖集中、不同 components | 区分真无共视与漏边；测试桥接候选、跨组 rematching、局部初始化。无可见桥接证据时不能强并 | COLMAP +强 matcher；GraphMatch；GLOMAP/GLUEMAP；同预算追加 pairs |
| **错误合并/折叠/错位拼接** | DG、Visym；Gendarmenmarkt | 不同图组映到同表面，视觉冲突、错误遮挡、少数桥边支配全局；残差也可能很低 | 验证实例/表面身份，定位异常 edges/tracks；局部 split、重验证、重新合并。不能把“模型更多组件”一律处罚 | DG++、MVDG、LFOE、Subgraph Pruning；传统 SfM 消歧 |
| **局部位置噪声及精度不足** | ETH3D Courtyard / PixSfM | 平面散点、短 tracks 不稳定、重投影分布异常 | 检查特征定位、几何条件与相机校准；featuremetric refinement、重新三角化/BA。当前没有明显 VLM 必要性 | PixSfM、Detector-Free SfM、固定 refinement |
| **全局平移/形状不稳定** | 1DSfM | 弱连接子图之间相对位置敏感、求解器多次结果不同 | 判断约束不足、错误平移方向或自洽错误；增加有效跨组约束、鲁棒平均或子图重估。不得仅靠“大楼看起来应对称”修正真实几何 | 1DSfM、GLOMAP、LFOE、GLUEMAP |
| **几何不完整** | ETH3D | 已注册但目标区域点云缺失、深度不一致 | 区分没有观测、纹理/细结构困难与错误 pose；前者不一定可修，后者可局部重新估计/密集重建 | 固定相机的强 MVS 与端到端系统分别比较，不把密集重建收益全归 Agent |

强基线来源：[GLOMAP](https://github.com/colmap/glomap)、[GLUEMAP](https://lpanaf.github.io/cvpr26_gluemap/)、[LFOE](https://openaccess.thecvf.com/content/CVPR2025/html/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.html)、[MVDG](https://arxiv.org/html/2610.01098v1)、[Subgraph Pruning](https://arxiv.org/html/2608.22054v1)。这些是建议纳入的候选基线，不代表每种方法的官方代码已经在上述每个数据集验证通过。

## 2026 更新：MVDG 并没有替代独立几何评测

[MVDG §4/5/10](https://arxiv.org/html/2610.01098v1) 在已有 DG/Visym 之外，从 AerialMegaDepth 的几何/相机信息产生 pseudo-pairwise 共视标签，作为高效训练与验证代理。该工作指出 DG 标注歧义，说明不能把所有 pair label 都视为无误。论文展示 Visym 场景中加入过渡视角能纠正两视图错误拼接；这是一条现有非语言多视图方法已具备的能力。

在没有额外审计 AerialMegaDepth 参考生成流程前，不应把它列为新的独立测量恢复 benchmark。其 pseudo-pairwise accuracy 与 SfM 指标的关系，也不能替代对我们目标错误的端到端评价。

## 对评测结论的约束

- **无一个上述常用集原生提供完整“失败原因→正确修复动作”标签。** 可以得到 pose/geometry/pair correctness，不能自动得到因果诊断 GT。论文明确的实例适合定义错误类型，不足以宣布“最频繁失败”。
- 参考 pose/深度、pair labels、geotags、独立 probe 注册结果属于 evaluator。测试时不给 Agent，也不用于接受/回滚；训练监督与测试反馈分开。公开下载得到的 test GT 同样不能作为推理输入。
- 固定输入图像池、相机内参假设与总预算；结果同时报告 pose/geometry、注册覆盖、component 处理及成本。只删掉难例以降低误差不能称恢复。
- IMC-PT、1DSfM 以及来自相同互联网地标的 DG 集合可能共享场景/图像；跨数据集泛化必须检查场景身份，不能只看数据集名称不同。
- 最有说服力的组合是 **DG/Visym 的自然歧义实例 + ETH3D 的测量几何约束**，但两者评测的是不同部分。不能预设 ETH3D 中自然存在大量同样的重复结构失败，也不能用人工注入错误替代自然失败结论。

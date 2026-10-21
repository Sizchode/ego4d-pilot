# 3D 重建错误修复的研究现状与切入点

Related-work survey｜2026-10-19｜范围：相机与几何修复

## 1. 一个具体错误

建筑转角两侧有两扇相似的窗。照片 A 拍左窗，B 拍右窗；程序误把窗角匹配为同一组三维点，连相机位置也一起算错。错误的相机和三维点仍可能互相吻合，因此投回图像的误差很小。假如已有照片 C 同时拍到两扇窗，并能认出 A 中的圆钟在左窗旁、B 中的铭牌在右窗旁，就有证据质疑这组匹配。问题是：怎样将这些图像线索落实为具体匹配的修正，并真正改善相机？这是示意例子，尚无实验结果。

## 2. 分开看错误类型和修复方法

错误类型回答“哪里出了问题”。对应不足：模糊、弱纹理或低重叠导致没有足够可靠匹配；错误对应：不同窗户被认成同一表面；模型假设不成立：物体运动、镜面反射或相机模型不合适；深度与融合错误：相机基本正确，表面仍有孔洞或误连接。同一个数据集可有多类错误，不能按数据集名称固定选工具。

修复方法回答“靠什么证据、改哪一步”。下表按主要机制分类；一篇论文可以跨类。局限指其方法范围，不等于这些方法在该类问题上都已失败。

| 方法类别 | 代表工作 | 算法核心 | 与本题的边界 |
| --- | --- | --- | --- |
| 几何诊断与恢复 | [Heinly 2014](https://www.cs.unc.edu/~jheinly/publications/eccv2014-heinly.pdf)；[Progressive SfM](https://arxiv.org/html/1803.07349v1) | 检测重建冲突；纠错或拆分、重算、合并模型 | 恢复和多步反馈早已有之；需证明新增图像证据的价值。 |
| 图像共视消歧 | [Doppelgangers++](https://arxiv.org/html/2412.05826v2)；[MVDG](https://arxiv.org/html/2610.01098v1)（2026 预印本） | 用两张或多张图判断是否看到同一表面，再筛选图像对 | 多视图上下文不是新意；图像对判断仍须验证最终相机。 |
| 学习过滤错误约束 | [LFOE](https://openaccess.thecvf.com/content/CVPR2025/papers/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.pdf)（CVPR 2025） | 结合相对几何、图上下文和视觉特征，识别错误图像对约束 | 图像与几何结合已有先例，不能说只有 VLM 会看图。 |
| 学习先验与几何融合 | [GLUEMAP](https://arxiv.org/html/2605.26103v2)（CVPR 2026） | 将前馈局部几何与全局相机求解结合 | 直接串接重建工具已有强基线。 |
| VLM 对象关联修复 | [ThinkGraphs](https://arxiv.org/html/2606.31471v1)（2026 预印本） | 已知相机位置下，判断并合并重复对象记录 | 修复对象记录，未证明能修复错误相机。 |

对应不足、低视差和深度融合，优先用匹配、几何与专门视觉方法；一般动态也已有 [RoMo](https://openaccess.thecvf.com/content/ICCV2025/html/Goli_RoMo_Robust_Motion_Segmentation_Improves_Structure_from_Motion_ICCV_2025_paper.html) 等方法。重复结构、动态和反光需要看图，但不因此就需要 VLM。本文只继续讨论重复结构的错误对应。

## 3. 我们可能补上的缺口

候选贡献只有一个：用可核查的图像证据，定位已影响相机求解的错误匹配。输出拟指向已有匹配列表中的具体点对，无需 VLM 生成像素坐标。与 Doppelgangers++、MVDG 的图像共视判断相比，目标要落实到具体错误关联及修正后的相机；与 ThinkGraphs 相比，相机本身也可能错误。更细的输出不是自动成立的创新，仍要证明它比现有消歧方法接入同一后端更有效。

VLM 可能有用的地方是把窗旁标识、周边布局和第三张图里的空间关系联系起来，而非替代几何求解。Agent 只有在继续选图、检查局部匹配后改变了修复决定，才有额外价值。照片来自同一批输入；模型提出可疑匹配与证据，几何程序验证后才执行修改。一次判断若同样有效，就不应强调多步 Agent。

尚未想清楚：怎样选待查匹配和下一张照片、怎样把对象区域对应到匹配点、何时修改及停止。不能仅凭重投影误差下降接受修改；如何引入独立观测验证，规则仍待设计。这些证据也可能被非语言模型利用。目前有研究假设，还没有成立的算法贡献。

## 4. Benchmark 与必须保留的 baseline

| 数据 | 要检验什么 | 指标与边界 |
| --- | --- | --- |
| [Doppelgangers / VisymScenes](https://arxiv.org/html/2412.05826v2) | 能否识别重复结构的错误关联 | 误接与误删；图像对标签不能直接当匹配点标签。 |
| [PhotoTourism](https://github.com/ubc-vision/image-matching-benchmark) | 是否真正修好相机 | 位姿误差、正确注册数；参考来自大规模重建，有自身偏差。 |
| [ETH3D](https://www.eth3d.net/data/schoeps2017cvpr.pdf) | 真实表面是否更准、更完整 | 激光参考下的几何指标；目标错误样本是否充足待确认。 |

系统对照：[COLMAP](https://demuc.de/colmap/)、GLUEMAP；消歧对照：Doppelgangers++、MVDG、LFOE 接入同一相机求解后端，具体接入需适配。不能只和默认 COLMAP 比，或只报告图像对分类。

控制器对照：相同图像、候选操作和求解器下，比较固定修复流程、非语言选择模型、一次性 VLM、多步 VLM。统一计算预算，并计入所有模型调用和失败尝试。只有多步 VLM 在相同预算下改善相机精度和正确注册覆盖，才支持 Agent 的价值。

评测缺口：仍需确定含自然错误且有可信相机参考的测试样本、正确恢复的误差门槛，以及具体匹配的诊断标注。参考答案仅用于离线评分。[DTU](https://roboimagedata2.compute.dtu.dk/data/text/multiViewCVPR2014.pdf)、[Tanks and Temples](https://www.tanksandtemples.org/uploadpreparations/) 的表面指标可补充几何评估，但不能直接当成相机修复成绩。

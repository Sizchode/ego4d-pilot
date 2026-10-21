# Benchmark-first failure taxonomy: MVS and non-Lambertian depth

核查日期：2026-10-19。范围：DTU、ETH3D、Tanks and Temples（T&T）；专项用Booster为主，3D Visual Illusion作互补。证据来自官方数据页、原始benchmark论文全文及方法官方论文。**下面的repair是可讨论的操作类别，不是已验证的Agent能力；历史论文观察不能直接当作2026系统的失败分布。**

## 1. Native task and reference provenance

| Dataset | 原始任务 / cameras | GT或reference来源 | 主要指标与边界 |
|---|---|---|---|
| DTU MVS | **Calibrated MVS**；提供高精度内外参。原始80-scene评测与网站124-scene全集、后续学习方法split不是同一个概念；49或64视角、7光照 | 工业机器人携带structured-light scanner；相机位置来自棋盘标定，不直接采用机器人指令位置；结构光reference points | Accuracy / completeness距离；使用observability mask。若隐藏外参改做SfM，应显式声明新协议，不能与使用GT cameras的MVS分数混比 |
| ETH3D high-res MVS | 已标定多视图；官方13train/12test。与ETH3D low-res视频、two-view stereo、SLAM任务分开 | Faro laser geometry；COLMAP初始化后通过scan-conditioned photometric alignment优化相机。提供的keypoints/triangulated points只是图像估计辅助数据，不是GT | 阈值accuracy、completeness、F1；unobserved space不计。激光参考与相机标定来源不同，后者不是独立motion-capture测量 |
| Tanks and Temples | **从video/images重建完整场景**，包含SfM；不提供精确intrinsics供固定使用。网站training下载COLMAP poses是baseline结果 | 工业Faro扫描、多位置注册清理；train公开geometry，intermediate/advanced测试geometry隐藏 | Precision/recall/F-score；提交geometry+相机轨迹，**官方不直接评价camera poses**，轨迹用于evaluation初始化。不能把training COLMAP poses当独立pose GT |
| Booster (TPAMI扩展版) | Calibrated balanced/unbalanced stereo及mono depth；606samples/85scenes，train228、stereo-test191、mono-test187。不是未知相机长序列重建 | 先拍自然外观，后对非朗伯表面喷涂/涂色并投影随机纹理，deep space-time stereo聚合生成reference disparities；不是直接用未处理玻璃的激光读数 | Bad-pixel、MAE/RMSE，按material mask分层；labels在半分辨率推得。原论文mono有GT scale/shift alignment；NTIRE2026 metric-mono是不同协议，不应混报 |
| 3D Visual Illusion | 单眼/双目depth；picture、replay、inpainting、holography、mirror | real72scenes/617frames用ZED+L515并有投影/补洞/过滤；大量training data来自web/generated videos与模型深度+plane correction | 适合视觉错觉depth和语义先验可靠性分析；不是完整camera recovery benchmark，也不是全部独立scan GT |

来源：[DTU官方](https://roboimagedata.compute.dtu.dk/?page_id=36)、[DTU标定说明](https://roboimagedata.compute.dtu.dk/)、[ETH3D文档](https://www.eth3d.net/documentation)、[ETH3D原论文](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/09/Schoeps2017CVPR.pdf)、[T&T下载](https://www.tanksandtemples.org/download/)、[T&T提交协议](https://www.tanksandtemples.org/uploadpreparations/)、[Booster原论文](https://arxiv.org/html/2301.08245v3)、[Illusion作者PDF](https://wu-yuwei-bit.github.io/paper/neurips2025_yao.pdf)。

## 2. Failures actually described by benchmark papers

| Failure family | 原论文明确证据（不是自然发生率） | 可观察diagnostic evidence | 可讨论repair与不能声称的内容 |
|---|---|---|---|
| Low texture → missing depth / holes | DTU2014 §5、Fig5：无纹理box出现孔洞；作者称lack of texture是**当时该组方法/数据**最大poor-performance来源。简单box meshing能补洞，复杂表面meshing会损失细节。ETH3D2017 §5.2/Fig8：office的弱纹理面，各方法completeness都低 | 相机支持正常但该区域patch ambiguity高、有效depth少；边界/邻域有可靠结构；需区分无观测与被filter删掉 | 换source views/scale、专用MVS或有证据的plane/normal先验、重新fusion。视觉“这是墙”不能直接证明平面/补洞正确；已有ACMM等处理低纹理 |
| Thin structures → lost geometry | ETH3D2017：COLMAP在electro、kicker、office、pipes的弱纹理/细结构困难；T&T Playground描述bars、poles、chains、beams，另有reflective slide | 多尺度depth边界、物体宽度相对pixel footprint、visibility/fusion删除轨迹 | 局部分辨率、细结构友好重建与fusion/meshing；需要GT检查completion和false positives。不是通常靠BA解决 |
| Wrong view selection / visibility support | ETH3D2017：Gipuma在courtyard、electro、delivery_area completeness低，论文归因object-centric view selection | 可见support数、baseline/overlap、source-view视角分布，depth估计与fusion分离日志 | 重选source views、局部重新估深度；几何策略已经直接适配，不能预设VLM必要 |
| Specular / non-Lambertian | DTU2014 Fig5：espresso pot高度specular；但全文观察是多视图整体受specularity影响**小于预期**，不能只截困难图。T&T Horse：bronze statue+uniform specular平台被明确称为挑战；Booster按material分层显示透明/镜面区误差更高 | view-dependent residual、mono/stereo disagreement、material masks、track provenance；固定mirror虚拟几何可一致 | 选未受反射污染视图、专用reflection modeling或有证据的depth融合。不能把所有镜面点视为pose outlier，也不能把material识别当新贡献 |
| Localization ambiguity from symmetry / weak background | T&T Francis：天空背景+雕像对称会增加camera localization困难；Palace：对称/重复、复杂多段轨迹和水面；**这是场景挑战描述，非归因统计** | 多个相对pose假设、匹配空间分布、跨帧连接与非重复支持 | 重匹配/扩展连接/跨视角验证/重估子图；必须与SfM消歧、图剪枝方法比，不应仅用MVS baseline |
| Photometric changes and low light | T&T Auditorium：uniform/repeating结构、剧烈照明变化和glare；Ballroom：弱光需要高ISO。DTU提供7光照，但2014原始评测仅使用近均匀照明 | 饱和/低信噪区域、跨视图appearance变化与几何支持分离 | 选择适用图像/曝光、鲁棒matching/photometric model；不能从数据存在低光推出是最大失败源 |
| Prior-driven fake geometry | 3D Visual Illusion：静态海报/屏幕内容误导单眼先验；mirror/glass可误导stereo | material/representation边界、parallax、plane支持、模型输出差异 | 调整可靠性/融合，但NeurIPS2025已用VLM融合mono/stereo；静态poster经典几何本可正确，不能称几何方法普遍不懂 |

具体原文：[DTU2014 §5/Fig5](https://roboimagedata2.compute.dtu.dk/data/text/multiViewCVPR2014.pdf)、[ETH3D §5.2/Fig8](https://www.microsoft.com/en-us/research/wp-content/uploads/2019/09/Schoeps2017CVPR.pdf)、[T&T §5.1–5.2](https://vladlen.info/papers/tanks-and-temples.pdf)、[Booster §V/material分层](https://arxiv.org/html/2301.08245v3)。DTU和T&T PDF通过作者站下载全文核查；本报告不把2014/2017结论延伸成当前模型排名。

## 3. Baseline selection follows the failure layer

| Claimed repair layer | 必须有的对照 | 比较时固定什么 |
|---|---|---|
| Dense MVS、低纹理/细结构恢复 | COLMAP MVS；ACMM（多尺度几何一致性与低纹理）；MVSFormer++、MVSAnywhere（较新学习型/跨域MVS） | 同一cameras、视图池、resolution与fusion；不能给新方法GT poses却让baseline估pose |
| Unknown camera / graph recovery | COLMAP/GLOMAP及当下强完整混合系统；task-specific graph filtering/disambiguation；完整输出再接同一MVS | 同一input/raw frames、matching工具、预算；geometry分数不能取代pose单独证据 |
| Stereo non-Lambertian | 原benchmark CREStereo/RAFT-Stereo用于历史追溯；当前比较至少包括Stereo Anywhere、FoundationStereo，和对应2026 challenge公开方法（使用同一赛道协议） | Calibrated stereo inputs、训练数据/finetune条件、full vs quarter resolution、material masks和GT alignment |
| VLM理解错觉/材料并纠正深度 | 3D Visual Illusion VLM fusion；专用segmentation+相同fix；非语言confidence/depth fusion | 相同底层depth/matching能力，隔离诊断/控制增量 |

这些是代表性强对照，不声明已穷尽截至2026-10所有SOTA。来源：[ACMM](https://openaccess.thecvf.com/content_CVPR_2019/html/Xu_Multi-Scale_Geometric_Consistency_Guided_Multi-View_Stereo_CVPR_2019_paper.html)、[MVSFormer++官方](https://github.com/maybeLx/MVSFormerPlusPlus)、[MVSAnywhere CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Izquierdo_MVSAnywhere_Zero-Shot_Multi-View_Stereo_CVPR_2025_paper.pdf)、[Stereo Anywhere CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Bartolomei_Stereo_Anywhere_Robust_Zero-Shot_Deep_Stereo_Matching_Even_Where_Either_CVPR_2025_paper.html)、[FoundationStereo](https://nvlabs.github.io/FoundationStereo/)、[NTIRE2026官方任务](https://cvlab-unibo.github.io/booster-web/ntire26.html)、[2026 challenge报告](https://iris.unive.it/handle/10278/5125410)。

## 4. Ground-truth blind spots matter to topic choice

1. **DTU reference is incomplete.** 原文明确有self-occlusion及underexposed structured-light holes，故使用observability masks。任何错误修复都要在可评价区域内讨论。
2. **ETH3D不是天然mirror benchmark。** 原文§3.2指出glass/reflecting objects引发laser系统离群；标定时mask掉透明表面、移动物体和不可可靠对齐区域。不能说laser GT自动解决镜面验证。
3. **T&T对全场景有效，不意味着每个反射区域都有精确GT。** 原文§3说明平面反射ghost可人工清除，复杂polished objects的少量spurious points可能保留；作者认为对总体评价可忽略。这不支持直接将这些局部区域当mirror recovery金标准。
4. **Booster用受控采集绕过材料问题，但任务更窄。** 涂层/纹理辅助reference有自己的处理假设与误差；它评价表面深度，不是反射世界多层geometry，也不是一般SfM恢复。
5. **3D Visual Illusion的真实与生成reference应分开。** L515也会遇到反射/透明问题；论文有后处理不能推出每个镜面像素独立真值。不得把20万训练图整体当真实测量GT。

## 5. What this permits us to claim

Benchmark evidence currently supports separating **pose/graph failures**, **dense-depth failures**, **fusion/meshing failures**, and **image-formation/learned-prior failures** before choosing an Agent problem. A low F-score alone does not identify the layer. Evidence is especially concrete for low texture, thin structures, view selection, and specific challenging scene conditions; mirror diagnosis has useful specialists but incomplete GT coverage in classic benchmarks. Nothing here establishes that mirrors are the dominant real-world failure or the best research direction.

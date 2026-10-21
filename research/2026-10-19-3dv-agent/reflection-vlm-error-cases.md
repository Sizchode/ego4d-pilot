# Mirrors, glass, pictures and screens: concrete failure cases and VLM limits

核查日期：2026-10-19。结论先行：这是有实际错误证据的一组问题，但“VLM认出镜子/屏幕，再mask或融合深度”已经有直接先例。剩余VLM增量必须是把语义解释绑定到具体几何约束、区分错误发生的层次，不能把recognition本身说成新贡献。以下修复建议是研究假设，未运行实验。

## Case 1: mirror produces a phantom room — geometry wrong does not imply cameras wrong

**具体场景。** 拍摄浴室/舞蹈教室，平面镜中出现另一组墙、桌子或人。重建在镜后产生一个额外房间，或把直接看到的实体和镜中实体混入同一feature track。前一种是虚拟几何被当作物理表面；后一种才可能把相机/结构联合优化拉错。

**为什么残差可能诊断不了。** 几何推导：固定平面镜中的静态物体有固定虚像位置；从移动的真实相机观察这个虚像，可以满足普通投影几何。因而低reprojection error与正确相机轨迹，完全可能同时伴随物理错误的镜后几何。不能宣称“镜面像素都随视角运动，所以一定是outlier”。直接/反射跨层对应、非平面镜、动态反射物体才会破坏这一简单模型。

**文献证据与近邻。** Reflect3r（首版2025-09-24，3DV2026）把镜面反射变成虚拟相机视图，使用对称约束优化，16个Blender场景带GT；它说明反射可以是有用几何信息，而非只能删除。Reflection-aware 3D mirror segmentation and pose estimation（online2024-11，卷期2025）用RGB-D共识、聚类和ray casting估计镜面，清除虚拟几何；其失败包含带深度的窗户被误认镜子以及强光造成边界错误。

**VLM可能的具体诊断。** 原图中“同一个人/家具及其反像”、镜框和遮挡关系，加上轨迹/track overlay，可能支持：该区域是反射，但相机是否已污染要另查；只修物理surface，或拆开direct/reflected tracks并重估受影响相机。看到镜子不是充分诊断。

**原子修复选择。** 给指定区域生成mask；隔离direct/reflected observations；用未污染tracks重新估计相关相机；或在有镜面模型时采用虚拟相机/反射约束。若mask后支持不足，不能宣称修复成功；用镜后点补全物理房间也不成立。

**最强对照与剩余增量。** 必须比较专用镜面检测器+固定恢复协议、RGB-D reflection-aware solver（模态适用时）、Reflect3r（只在其单图+反射设定可比），而不是只和COLMAP比。潜在增量是区分“虚拟geometry但pose正确”和“跨层错误tracks导致pose错误”，并把该判断落实到恰当干预；本轮未找到证明VLM胜过这些专用方法的证据。

**反例。** 专用mask足够解决的镜面场景，VLM多步理解没有必要；窗外真实场景不能因看似镜面而删除；不在直接视野出现的反射物体不提供可靠identity线索。

## Case 2: screen/poster becomes a fake 3D scene — learned prior failure, not necessarily SfM failure

**具体场景。** 房间墙上有立体感海报/大电视，屏幕播放道路或室内视频。单眼或前馈重建可能把海报中的深度当作真实凸起/洞口，把电视内容当作可进入的空间。若屏幕内容变化，视频内部纹理运动也可能与真实相机运动混淆。

**应分清三件事。** 静态poster和静态screen内容是实际平面上的稳定纹理，足够视差的经典matching本可正确恢复平面；它们不是普遍的SfM难例。播放中的screen内容才违反静态纹理对应，须分开屏幕外壳/边框和内部显示。固定平面镜则呈现虚拟3D，不能用screen平面逻辑处理。

**直接近邻，不能漏掉。** NeurIPS2025《3D Visual Illusion Depth Estimation》已经提出pictures/replay/holography/mirror等类别，并用VLM输入RGB、monocular depth、stereo disparity来预测置信度及融合。它明确指出texture illusions偏向骗过单眼先验，而mirror/glass偏向破坏常规stereo解释。故“VLM识别画中画→纠正depth”已有论文，不能claim空白。

**VLM可能的具体诊断。** 模型结合可见屏幕边界、UI/字幕、跨帧内容替换，以及点/track落点，提出“这段错误约束来自显示内容，而非真实相机剧烈运动”；对于静态poster应保留稳定特征并拒绝单眼深度先验，对于dynamic screen应排除内部时变tracks而保留外框，属于不同修复。

**原子修复选择。** 检验指定区域的单应/多视角track；对实际支持平面的区域降低错误单眼prior权重；删除被证实属于时变内容的observations；重估受影响相机。不能仅因为对象标签是screen而一律mask，更不能只凭语言断言平面。

**最强对照与剩余增量。** 上述VLM融合方法、专用display/illusion detector+相同工具、动态几何/多模型运动分割、homography+同一SfM backend。剩余可能是诊断已发生的多视角相机错误、找到传播到哪些tracks/cameras、针对性回滚重算。它不是单幅深度融合的新版本；screen导致真实SfM失败的发生率和恢复收益本轮未核实，不能据此优先立题。

**反例。** 海报是静态纹理时几何可能比VLM可靠；没有边界的屏幕或刻意trompe-l'oeil可能同样骗VLM；“像屏幕”的真实窗口是困难负例；播放内容单独形成一致的另一段视频时，语义推断仍需外部静态区域验证。

## Recent specialist landscape

| Work | Evidence checked | Consequence for our novelty |
|---|---|---|
| [Reflect3r](https://arxiv.org/html/2509.20607v2) | Full HTML method/experiments | Reflection exploitation and symmetric pose refinement already exist; single-view setting is not generic SfM recovery. |
| [Reflection-aware 3D mirror segmentation and pose estimation](https://link.springer.com/article/10.1007/s00371-024-03704-9) | Full paper including limitations | Dedicated geometric diagnosis already resolves some reflective ambiguity, with explicit window failure cases. |
| [Detect Any Mirrors, CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Xing_Detect_Any_Mirrors_Boosting_Learning_Reliability_on_Large-Scale_Unlabeled_Data_CVPR_2025_paper.html) | Official abstract + author code; PDF snippets | Iterative data engine already includes multi-modal agent semantic scoring for mirror pseudo-labels; not online SfM repair. |
| [MirrorSAM, AAAI2026](https://winter-flow.github.io/project/MirrorSAM/) | Author project abstract | Adapts SAM with specialist mirror modules; compare specialist, not only vanilla SAM. |
| [MirrorSAM2](https://arxiv.org/abs/2509.17220) | Abstract | Prompt-free RGB-D video mirror segmentation, VMD/DVMD; strong temporal specialist but input modalities differ from RGB SfM. |
| [SVRNet, TCSVT2026](https://winter-flow.github.io/project/SVRNet/index.html) | Author page and linked paper screened | Scene-aware visual reasoning/prompt chain for mirrors is already a specialized non-conversational architecture. |
| [3D Visual Illusion Depth Estimation, NeurIPS2025](https://wu-yuwei-bit.github.io/paper/neurips2025_yao.pdf) | Downloaded author PDF, methods and data sections read | VLM already consumes RGB+geometric predictions to assess their reliability; broad novelty claim contradicted. |

## Benchmark applicability, not a ready-made camera-recovery test

- Reflect3r: synthetic16scenes have independent rendered geometry/poses, but designed for single-view mirror stereo, not multi-view failure recovery.
- MSD/PMD/RGBD-Mirror and VMD/DVMD: masks/segmentation validation, not sufficient for camera-recovery outcome.
- 3D Visual Illusion real subset: 72scenes/617frames, ZED Mini+L515; paper applies upsampling, projection, hole filling, consistency rejection and filtering. This is depth/disparity evaluation, not verified multi-view camera GT. Training largely web/generated data, monocular/DUSt3R-derived depth with plane correction; do not call the full dataset independent scan GT. Reflective material also complicates sensor truth.
- Booster, used by illusion paper, is a relevant specialist stereo comparison; exact official protocol needs separate audit before use.

## Research judgment

The concrete remaining hypothesis is narrower than “VLM understands mirrors”: **can image semantics plus geometric provenance distinguish an image-formation explanation from an actual camera/track failure, then select a materially different repair?** These cases make that hypothesis intelligible, but do not establish a VLM advantage. The retrieved closest work already performs VLM-conditioned depth reliability/fusion; proving superiority to detectors and geometry-aware specialist controls is essential. A mirror-only task would likely place most novelty in the specialized geometric tool, not the agent.

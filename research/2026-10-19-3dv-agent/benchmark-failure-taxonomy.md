# Benchmark-first reconstruction failure taxonomy

调研日期：2026-10-19。范围：真实多视图 SfM/MVS、室内视频建图、对象级重建；专项数据用于补充重复结构、动态与成像退化。不是所有 3D benchmark 的穷举。

## 结论与证据标准

先按 benchmark 的真实困难与已有失败来定义诊断任务，再确定证据、工具和控制策略，是合理的研究顺序。需要区分三个层次：数据条件（如低纹理）、失败机制（如没有足够可靠对应）、输出症状（如未注册/孔洞）。同一个数据集、同一个症状可以涉及多个机制。

本文没有运行新实验，也没有跨算法的统一失败频率统计。因此“典型错误”表示原论文报告的失败或有明确依据的风险，不代表2026年所有模型的最高频错误。修复操作和VLM作用是本文分析；不假定操作一定有效。

## 1. 常用 benchmark 与主要诊断对象

| Benchmark | 原始任务与参考 | 优先调查的错误 | 能证明什么与协议边界 |
|---|---|---|---|
| DTU | 标定多视图、结构光表面参考；机器人重复采集、多光照 | 无纹理表面孔洞；高光区域错误深度；复杂结构缺失 | 原生标定 MVS 可评价 accuracy/completeness/overall。不能把已提供外参的成绩解释为相机恢复能力；隐藏外参是另一个明确声明的协议。 |
| ETH3D | 室内外、激光表面参考；高分辨率少视图与低分辨率多视图分别评测 | 弱纹理/细结构不完整；视图选择与相机布局不适配；稀疏观测约束不足 | 官方几何 accuracy/completeness/F-score；固定外参与重新估计外参分开报告。 |
| Tanks and Temples | 真实大场景视频，工业激光参考 | 对称/重复结构造成注册歧义；均匀反光材质、细结构和遮挡造成表面缺失 | 官方主结果是几何 F-score；提交相机用于初始化对齐，官方明确不直接评价位姿。 |
| PhotoTourism / Image Matching Benchmark | 非受控地标照片，宽基线、光照差异；大规模 COLMAP 重建产生伪参考 | 小图集对应不足、匹配失败、相机未注册；重复结构可能引入错误对应 | 相对位姿与多视图准确率，按具体版本使用 mAA 等指标；不要混用不同年份 challenge。 |
| 1DSfM | 互联网照片的全局 SfM；Bundler 参考 | 错误相对平移方向、跨子图连接不足；自洽错误边可能逃过过滤 | 可比较全局 SfM 鲁棒性，但参考不是独立测量真值。 |
| ScanNet / ScanNet++ | 室内 RGB-D；ScanNet 为重建系统产生的姿态/表面，ScanNet++另有注册激光扫描与高质量图像 | 遮挡、弱纹理、细小结构；适合检查对象关联与几何冲突，具体发生率待审计 | ScanNet++的高质量几何有价值，但官方 NVS/语义任务不是现成相机修复任务；NVS test 不公开扫描。不能当动态场景GT集。 |
| CO3D | 物体环绕视频；相机/点云为自动 SfM/MVS 标注 | 少视图或宽基线子集下重叠不足、相机估计失败；背景/前景分离与遮挡风险 | 已有少视图pose协议可用；原生challenge为已给相机的NVS。参考有重建成功筛选偏差，不是独立扫描真值。 |
| TUM RGB-D dynamic / Bonn RGB-D Dynamic | 动态室内RGB-D，外部运动捕捉相机轨迹；Bonn另有独立静态环境激光点云 | 动态前景被用于估计相机；初始动态物体进入静态地图；大面积遮挡导致跟踪失败 | ATE/RPE与轨迹完整率；Bonn还可评价静态地图。动态序列不是完整4D表面GT。RGB-only与RGB-D需分别公平比较。 |

主要来源：[DTU官方](https://roboimagedata.compute.dtu.dk/?page_id=36)、[DTU原始评测](https://roboimagedata2.compute.dtu.dk/data/text/multiViewCVPR2014.pdf)、[ETH3D数据格式](https://eth3d.ethz.ch/documentation)、[ETH3D评测工具](https://github.com/ETH3D/multi-view-evaluation)、[Tanks and Temples数据](https://tanksandtemples.org/download/)、[Tanks and Temples评测边界](https://www.tanksandtemples.org/uploadpreparations/)、[Image Matching官方代码与协议](https://github.com/ubc-vision/image-matching-benchmark)、[ScanNet++文档](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation)、[CO3D challenge](https://github.com/facebookresearch/co3d/blob/main/co3d/challenge/README.md)、[TUM序列说明](https://webshare.cvg.cit.tum.de/g/rgbd/dataset/)。1DSfM、Bonn及逐场景定位见本目录的benchmark-taxonomy-sfm.md和benchmark-taxonomy-video.md。

## 2. 应补入的专项集

| Benchmark | 为何需要 | 主要局限 |
|---|---|---|
| Doppelgangers / VisymScenes | 直接研究重复/对称建筑的错误共视关系和错误合并；比仅凭普通数据集低分推测身份错误更明确 | 图像对、地理位置或歧义标注不等于独立精密的逐相机/表面GT。正确拆开仍需计算覆盖损失。 |
| RealX3D | 低光、光照变化、烟雾、暂态遮挡、反射和模糊；干净/退化观测与激光几何能支持分类型评价 | 参考相机由干净图像COLMAP得到；相机恢复时不能把它作为输入。motion blur分支由渲染积分合成。玻璃反射层不等同镜面房间。 |
| MVM-IOD | 工业物体、不同背景；标定参考相机与测量扫描，明确显示传统SfM和前馈模型的不同失败 | 小规模工业专项；图像格式/朝向影响是分布敏感性证据，不应写成针对数据集的硬编码修复规则。 |
| Booster | 透明/镜面等非朗伯材料专项；通过材料处理与投影纹理辅助获得视差参考 | 已标定stereo/mono depth，不能直接验证未知相机SfM恢复；具体参考处理与赛道见benchmark-taxonomy-mvs.md。 |

来源：[Doppelgangers++](https://arxiv.org/html/2412.05826v2)、[RealX3D](https://arxiv.org/html/2512.23437v1)、[MVM-IOD](https://arxiv.org/html/2606.16638v1)。

Booster来源：[原论文](https://arxiv.org/html/2301.08245v3)。详细子报告：[SfM与重复结构](benchmark-taxonomy-sfm.md)、[MVS与材质](benchmark-taxonomy-mvs.md)、[室内视频与动态](benchmark-taxonomy-video.md)。

## 3. 统一 taxonomy：诊断对象、证据与干预

这些类别是诊断工作分类，允许多标签；不是互斥且已经有频率标注的官方 taxonomy。

| 类别 | 需要区分的机制与症状 | 主要数据 | 推理时可取得的证据 | 候选干预 |
|---|---|---|---|---|
| F1 可靠对应不足 | 低纹理/低光/模糊导致可用特征不足；未注册、短tracks、表面孔洞 | DTU、ETH3D、RealX3D | 原图局部质量、匹配空间覆盖、跨帧稳定性 | 更合适的匹配方法或分辨率；使用已有清晰帧；重新验证恢复图像上的匹配 |
| F2 对应身份错误 | 不同表面/实体因重复图案被误匹配；错误loop、模型折叠/合并；残差可能不高 | VisymScenes、PhotoTourism、T&T | 匹配所在区域、额外视图、共视/不共视证据、独立pair几何 | 重验/删除错误edges，拆分错误tracks，再重估相关相机 |
| F3 观测几何不足 | 小视差、低重叠、缺桥接图像或跨子图支持；深度不稳定、components断开 | CO3D少视图、1DSfM、ETH3D | 三角化角、有效共视关系、独立相对位姿、尚未使用的图像 | 在给定图像池中找桥接或有视差视图；重新选初始子图；无法观测时保留不确定性 |
| F4 单一静态场景假设失败 | 物体运动/状态变化污染相机或静态地图；运动前景可能低残差 | TUM/Bonn、RealX3D | 时间序列、对象级流、前景与静态背景分别拟合的支持 | 识别实际运动区域、重新估相机、分时段/分运动模型重建 |
| F5 成像/材质模型失配 | 高光、反射、透明引起不对应真实表面的深度或虚拟几何 | DTU、T&T、RealX3D与镜面专项 | 多视图区域外观变化、深度分歧、材质与反射边界 | 标记不可靠观测、选择适合观测/模型；只在约束受污染时重估相机 |
| F6 相机/全局估计错误 | 内参/畸变不适配，错误初始化、漂移或局部解；整片几何系统性变形 | PhotoTourism、1DSfM、ScanNet++、MVM-IOD | 残差空间分布、内参稳定性、独立相对位姿、替代初始化结果 | 正确相机模型及参数共享；重初始化、局部/全局优化；不能依靠BA单独解除错误身份 |
| F7 深度/融合阶段错误 | 相机基本可靠但遮挡边界、细结构、弱纹理深度缺失，或融合误删/连接 | DTU、ETH3D、T&T | 每视图深度与融合结果分开检查，深度支持数和可见性 | 调整source-view选择、重估局部深度、重新融合；不因表面孔洞盲目重估所有相机 |

VLM可能提供的增量主要集中在F2、F4、F5中的视觉解释，以及同一症状下的阶段归因。F1/F3/F6/F7不能默认需要语言模型；应先与直接使用同一图像/几何信息的专门模型比较。

## 4. 用论文中的具体例子检查 taxonomy

- **DTU**：2014原始评测将缺乏纹理列为其被测方法主要表现不佳的来源；无纹理盒子出现孔洞，高光espresso pot也困难。这是该研究结果，不能扩成所有现代方法的统计结论。[原论文](https://roboimagedata2.compute.dtu.dk/data/text/multiViewCVPR2014.pdf)
- **ETH3D**：原论文报告COLMAP在弱纹理与细结构上的缺失；electro/kicker/office/pipes是相关例子；Gipuma在courtyard/electro/delivery_area的完整度问题与偏object-centric的视图选择有关。具体原文定位见benchmark-taxonomy-mvs.md。
- **T&T**：Horse涉及均匀高光材料；Playground有杆、链和反光滑梯；Francis的天空背景与对称性增加定位难度。这些是作者的场景挑战描述，并非某个方法统一统计出的失败频率。原论文见benchmark-taxonomy-mvs.md。
- **1DSfM**：Gendarmenmarkt中跨建筑宽基线导致连接少，困难不只是重复穹顶误匹配；这是F2与F3需要区分的例子。原文及Bundler参考限制见benchmark-taxonomy-sfm.md。
- **TUM**：ReFusion讨论walking_xyz开头相机随人运动，人物进入模型而没有后续无遮挡重访来纠正；“运动物体始终高残差”不是可靠假设。原文定位见benchmark-taxonomy-video.md。
- **MVM-IOD**：作者报告部分绿色背景场景COLMAP失败、前馈模型在原始倒置方形图像上产生多份不同方向物体。两种输出失败属于不同机制；关于分布失配的原因是作者解释，不能据此建立确定性标签。[§4](https://arxiv.org/html/2606.16638v1)

## 5. 由分类得到控制问题，而不是按数据集写死规则

以“注册率低”为例，它可能来自F1无可靠特征、F2误匹配、F3缺少桥接或F4动态污染。诊断应先定位失败图像/边/区域，再找能区分这些假设的证据；不同假设得到不同的干预对象。

一个具体episode可以是：失败子图包含两个相似立面；检查匹配原图与第三视角，判断是假共视还是漏了桥接；前者删/拆错误约束，后者增加已存在图像池内的连接；重新执行固定后端；再检查独立观测支持与整体覆盖，而不是只看训练残差下降。

数据集名称、scene名称和离线错误标签不作为policy输入。官方固定输入协议不允许额外帧时，追加帧应作为单独的pool-recovery协议，不能混报官方结果。真实新视角采集又是另一任务。

## 6. Baseline必须按失败类别配置

| 层级 | 对照 |
|---|---|
| 完整后端 | COLMAP incremental/global mapper、前馈基线，以及合适的hybrid SfM；版本、匹配器与输入预算固定 |
| F1/F7 | learned matching、多尺度MVS、source-view选择和一致性融合；例如ACMM/PatchmatchNet类别，不把简单调参作为唯一对照 |
| F2 | Doppelgangers++、LFOE、instance/re-ID或共视验证；固定相同后端 |
| F3/F6 | 扩展匹配、固定重初始化/BA流程、鲁棒全局SfM与混合几何方法 |
| F4 | DynaSLAM/ReFusion类RGB-D方法（同传感器协议），RoMo+SfM；VLM近邻Dynapo |
| F5 | 专用材质/镜面检测与适当几何模型；VLM辅助深度方法也要纳入 |
| 控制器 | 同一toolbox下固定流程、一次性选择、非语言学习型选择、多步VLM；独立核算所有尝试成本。GT选择动作只作为离线oracle。 |

较新的完整基线尤其不能漏：[GLUEMAP](https://github.com/colmap/gluemap) 已将检索、Doppelgangers++共视、前馈局部几何、全局优化和轨迹精化组合起来。GLOMAP本身已[迁入COLMAP global mapper](https://github.com/colmap/glomap)。把这几项工具串接起来不是新贡献。

专项参考：[ACMM](https://openaccess.thecvf.com/content_CVPR_2019/html/Xu_Multi-Scale_Geometric_Consistency_Guided_Multi-View_Stereo_CVPR_2019_paper.html)、[RoMo](https://openaccess.thecvf.com/content/ICCV2025/html/Goli_RoMo_Robust_Motion_Segmentation_Improves_Structure_from_Motion_ICCV_2025_paper.html)、[DynaSLAM](https://arxiv.org/abs/1806.05620)、[LFOE](https://openaccess.thecvf.com/content/CVPR2025/papers/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.pdf)。

## 7. 从survey到研究定位

这张taxonomy给出了错误覆盖与可验证性的地图，还没有给出统一失败分布、每类恢复上限或VLM优势。现阶段可以先选定研究输出：相机/图修复优先PhotoTourism、1DSfM、重复结构专项；密集几何优先ETH3D、T&T、DTU；动态诊断用TUM/Bonn；真实退化用RealX3D补充。不要一开始把所有类别和后端变成同一篇论文的承诺。

如果延续实例身份→相机约束修复，最清楚的数据组合是重复结构专项暴露F2，再用具有独立几何参考的数据检查修复的下游后果；专项集能否提供所需feature-track诊断标签仍需核对。ScanNet++的实例注释与激光几何提供另一条室内研究路径，但需要明确定义未知相机协议和真实失败样本，而不能把已有语义榜单直接当恢复榜单。

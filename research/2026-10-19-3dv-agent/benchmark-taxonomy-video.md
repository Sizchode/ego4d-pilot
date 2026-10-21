# 从 benchmark 出发的重建错误 taxonomy：室内视频、动态场景与 CO3D

核查日期：2026-10-19。仅整理原始文献与官方数据说明，没有运行方法。**数据存在某种挑战，不等于某个算法在全数据集上必然失败；可移动类别不等于拍摄期间实际运动；语义标签不是运动 GT。**

## 1. 数据集与可支持的主张

| Dataset | 原生任务与输入 | 参考 pose / geometry 来源与局限 | 对 recovery 的合理用途 |
|---|---|---|---|
| **TUM RGB-D dynamic subsets** | RGB-D SLAM/odometry；sitting、walking、desk_with_person 等，含人物/对象运动和不同相机运动 | 外部 100 Hz motion capture 轨迹；输入 depth 是消费级传感器观测，不能当独立高精度完整静态表面 GT；部分序列 GT 有时间缺口 | 动态污染导致的 tracking loss、错误背景估计；主要看 ATE/RPE、轨迹覆盖/丢失时间，明确 RGB-only 或 RGB-D |
| **Bonn RGB-D Dynamic** | 动态 RGB-D SLAM；24 dynamic + 2 static 序列，crowd、balloon、moving/placing/removing boxes、person tracking | OptiTrack Prime 13 camera GT，另有 Leica BLK360 **静态环境**点云；需遵守官方 sensor/marker/ROS frame 校准；没有因此获得完整动态对象4D GT | 本组最完整的 pose + static geometry recovery 证据；可区分遮挡与非遮挡动作、人体与搬动物体 |
| **ScanNet** | 大规模室内 RGB-D 扫描与语义场景理解；社区另用于 tracking/reconstruction | 官方使用 BundleFusion 生成 camera trajectory / mesh，属于同源重建参考；不是外部 mocap 与独立激光真值 | 低纹理、覆盖、跟踪/融合鲁棒性及跨场景泛化；不能称官方动态恢复 benchmark |
| **ScanNet++ v2** | 当前 1006 场景；DSLR/iPhone/laser；官方 NVS、语义和实例任务 | laser point cloud/mesh 是独立几何测量来源；DSLR COLMAP camera model 对齐 laser；iPhone 有原始/对齐 ARKit 与筛选后的 COLMAP poses，不能混用；mesh 渲染深度属于派生参考 | 手机视频质量、配准、静态几何恢复；适合独立几何检查，但 camera reference 不是 mocap。**没有核实到逐像素实际 motion GT 或动态恢复官方任务** |
| **CO3D** | 物体类别重建/NVS；采集要求静态物体放稳，拍摄者绕一圈；不是转台动态对象 benchmark | cameras、MVS depth/point clouds 来自 COLMAP，mask 来自 PointRend；原论文过滤不准相机和点云；不是外部测量真值 | 物体多视图、稀疏输入和跨类别泛化；能衡量与参考解一致性，不能单独证明超越原COLMAP几何的真实正确性 |
| **Wild-SLAM（补充）** | CVPR25 WildGS-SLAM 的 MoCap 和 iPhone 两子集；动态 distractors | MoCap：10 RGB-D序列、OptiTrack、仅2静态布局；iPhone：7自然序列，论文明确无GT trajectory，仅定性使用 | 扩展到机器人、桌子、伞、球等实际运动；不能把 iPhone 定性 failure 当作已有可量化 pose GT 案例 |

官方来源：[TUM](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)、[TUM sequence descriptions](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)、[Bonn](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/)、[ScanNet 官方仓库](https://github.com/ScanNet/ScanNet)、[ScanNet++ documentation](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation)、[CO3D 原论文](https://arxiv.org/html/2109.00512v1)、[WildGS-SLAM](https://arxiv.org/html/2504.03886v1)。

## 2. Benchmark-grounded error taxonomy

下表严格区分 **E：作者报告的具体 failure**、**C：官方列明的 challenge** 与 **H：尚待验证的诊断/修复假设**。修复列不表示已验证可恢复。

| 错误类型 | Benchmark 与已有证据 | 诊断时该看什么 | 候选修复动作与边界 |
|---|---|---|---|
| **相机跟随运动主体，主体被错误融合成背景** | **E** ReFusion 的 TUM `walking_xyz`：开头跟随人物，前4秒相对位移误差高；该位置未重访，人物残留静态模型 | 人体/背景的独立 tracks、motion residual、静态背景观测历史、被融合区域的原始帧 | **H** 回溯分离主体约束，利用剩余背景重估pose并反融合错误区域；没有观测的背景不能凭空补成正确几何 |
| **动态物体/遮挡占据初始化阶段，丢失跟踪** | **E** ReFusion 报告 StaticFusion 在 TUM `walking_halfsphere` 开头因动态元素过多失跟踪；**C** Bonn obstructing-box、crowd | 初始静态支持区域是否足够、候选多运动分量、mask后track覆盖、初始化可观测性 | **H** 使用之后的可靠帧初始化再回溯、重匹配稳定区域；若整段无背景或无视差，应拒绝世界pose恢复而非硬输出 |
| **短暂停止运动/缺少无遮挡背景，低残差误当静态** | **E** WildGS-SLAM `Wandering`：人物在部分帧不动，拍摄方向相近、无早期干净背景，uncertainty过低，人物留在map | 整段运动史而非单帧类别，遮挡与可见性、静态解释是否依赖从未独立验证的区域 | **H** 对短暂停留对象保留时间状态，追踪更长时间或引入已观察静态区域；VLM可提出语义先验但不能以“人可动”替代运动证据 |
| **静态复杂外观难拟合，被不确定性模块误当应忽略区域** | **E** WildGS-SLAM `Shopping`：复杂墙纹理难重建→高SSIM loss→错误静态uncertainty | 光流/几何一致性与photometric loss是否冲突、静态区域不同视角的持续性 | **H** 区分外观建模失败与真实动态，避免删除墙面；调图像/外观模型或几何验证。该论文两个failure是iPhone定性结果，无GT trajectory |
| **低纹理+较大相邻视角位移，特征/关联不足** | **E** SplaTAM 报告其使用的两条ScanNet++序列中Point-SLAM/ORB-SLAM3跟踪失败，ORB-SLAM3因缺特征多次重初始化；**C** ScanNet++原论文强调大无纹理墙/柜体 | 匹配空间覆盖、实际parallax、depth约束与色彩特征支持、图连通性；不只看匹配总数 | **H** dense/depth-assisted matching、增加中间帧、重注册/局部优化；该证据不能推广成ScanNet++全部场景都失败，也不自动需要VLM |
| **运动模糊、视场/相机模型或深度质量引起不一致** | **C** ScanNet++原论文说明ScanNet的motion blur/limited FOV；官方ScanNet++提供fisheye/undistorted版本、`is_bad`（blur/heavy shadow）、ARKit和COLMAP多种pose；**E** ReFusion讨论depth零值混合了out-of-range和不可测量 | 清晰度、曝光、正确内参/畸变/缩放、RGB-depth对齐、invalid-depth分布；不把0当可用深度 | **H** 正确校准/解畸变、选邻近清晰帧、对无效深度作明确处理；配置错误和不可观测损失应分开，不靠VLM发明深度 |
| **稀疏物体视角无法注册/几何缺失** | **C** CO3D原任务研究少量source views，原始COLMAP生成过程承认tracking和MVS可失败；**没有**逐类别“重复纹理/反光导致失败”标签 | 目标与背景的feature support、overlap/parallax、mask质量、注册率、相机/点云质量标记 | **H** 选择补充已有视频帧、改匹配/初始化、重跑MVS；人工子采样引起的失败要明确叫受控setting，不冒充原生自然失败 |

具体失败出处：[ReFusion Sec IV-A/B、Fig7–8](https://www.ipb.uni-bonn.de/pdfs/palazzolo2019iros.pdf)、[WildGS-SLAM supplement Failure Cases / Fig12](https://arxiv.org/html/2504.03886v1)、[SplaTAM camera-pose evaluation](https://openaccess.thecvf.com/content/CVPR2024/papers/Keetha_SplaTAM_Splat_Track__Map_3D_Gaussians_for_Dense_RGB-D_CVPR_2024_paper.pdf)、[ScanNet++ ICCV23正文](https://openaccess.thecvf.com/content/ICCV2023/papers/Yeshwanth_ScanNet_A_High-Fidelity_Dataset_of_3D_Indoor_Scenes_ICCV_2023_paper.pdf)。

## 3. Baselines 应由错误与信息条件决定

### TUM/Bonn：动态 pose 与 static geometry

- RGB-D 传统强对照：**DynaSLAM N+G、ReFusion**；前者结合语义和几何，后者基于几何残差与free-space处理。不能只对普通COLMAP。
- RGB-only现代对照：**WildGS-SLAM、MegaSaM、MonST3R**；WildGS已在TUM/Bonn报告相机结果，使用DINOv2不确定性支持tracking/mapping。其MonST3R-SW是作者滑窗适配，不应等同原始完整batch模型。
- 直接VLM对照：**Dynapo masks + 同一个固定pose/refinement backend**，它已经用VLM运动推理改善几何；其原论文TUM实验是depth优化，不能误写成已报告TUM camera benchmark。
- 2026动态重建机制对照：**Flow4R、SM4RT**，分别有scene-flow/参考系权重、结构化motion bases；本轮未核实它们在TUM/Bonn全套协议上的结果，属于应适配的候选而非现成同协议leaderboard。
- 若目标包含动态对象几何保留，补 **4D Gaussian Splatting SLAM（ICCV25）**：其TUM/Bonn任务保留动态，与仅恢复静态背景不同；部分中途出现动态对象场景需要预指定初始化帧，公平比较需说明。

来源：[DynaSLAM](https://arxiv.org/abs/1806.05620)、[WildGS-SLAM](https://openaccess.thecvf.com/content/CVPR2025/html/Zheng_WildGS-SLAM_Monocular_Gaussian_Splatting_SLAM_in_Dynamic_Environments_CVPR_2025_paper.html)、[MegaSaM](https://mega-sam.github.io/)、[Dynapo](https://arxiv.org/html/2512.05398v1)、[Flow4R](https://arxiv.org/html/2602.14021v1)、[SM4RT](https://arxiv.org/html/2607.22534v1)、[4DGS-SLAM](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Li_4D_Gaussian_Splatting_SLAM_ICCV_2025_paper.pdf)。

### ScanNet / ScanNet++：静态配准与几何

RGB-D在线对照包含 **ORB-SLAM3、SplaTAM**；可补同传感信息条件下当前完整SLAM系统。若设为离线RGB恢复，应比较 **COLMAP/GLOMAP + 固定matcher** 与 **VGGT/GLUEMAP等强初始化/重建方案**，并说明这已是自定义unposed protocol，而不是官方NVS排行榜。

把参考深度/laser只用于评估与把它们交给算法是两个不同setting。DSLR fisheye、undistorted与iPhone要分清；不能把几何GT暴露给诊断器再称GT-free recovery。

### CO3D：物体多视图与稀疏pose

同输入的COLMAP（强matcher）、VGGT/MASt3R及合适的pose方法才是对照；**NerFormer/NeRF等给定相机的原生NVS baseline不能直接当未知pose恢复baseline**。精简数据时先固定CO3Dv1/v2、类别/sequence split与可用帧数，避免把不同过滤版本混在一起。

## 4. 必须保留的评测边界

1. **静态/动态标签不是原因标签。** 每个episode需要注明实际观测到的运动，而不是按person/car/chair类别直接生成error GT。
2. **GT来源决定结论强度。** Bonn可用独立静态激光和mocap；TUM重点pose；ScanNet/CO3D是重建派生参考；ScanNet++独立geometry与估计camera需分开写。
3. **避免成功者偏差。** CO3D原论文剔除18%不准tracking视频，测试候选还要求point-cloud检查通过；不能用这个集合声称覆盖原始COLMAP所有自然失败。这是v1论文生成流程，不推定v2比例相同。
4. **轨迹准确与轨迹覆盖并报。** 单报成功帧ATE会奖励丢弃困难帧；需要注册/跟踪覆盖、failure rate、recovery latency及总compute。RGB-only可允许单一Sim(3)对齐，RGB-D尺度已知时应说明对齐协议。
5. **“删干净”不等于“重建好”。** 静态地图accuracy必须配completeness；动态区域去掉后留下洞、把复杂静态墙删掉，都不能靠低误差掩盖。
6. **常用集并不天然证明VLM增量。** TUM/Bonn有人体偏置；ReFusion作者明确承认Bonn每条动态序列都有人的偏置。需要同mask/同工具对照，排除收益只是更强分割或更大后端。

## 5. 最值得放进主survey的排序

- **动态恢复主证据：TUM + Bonn**；Bonn补静态geometry，TUM补常用pose协议。
- **现代动态泛化补充：Wild-SLAM MoCap**；iPhone failure用于提出机制问题，不能替代定量pose结果。
- **一般手机/室内恢复：ScanNet++，ScanNet用于广度**；不能把它们误归为dynamic GT benchmark。
- **物体/稀疏视角泛化：CO3D**；不是动态或原始失败恢复的主benchmark。

当前最具体的研究观察是：**同样的高残差可能是动态、外观难拟合或pose错；同样的低残差可能来自正确静态结构，也可能是相机跟随的人。** 这有真实文献failure支持，但不自动证明VLM能可靠区分，且Dynapo等已经覆盖相当部分语义增量。

# GRASE 与我们的任务边界

**结论。** 两者最相近的不是输出，而是控制方式：都让 training-free VLM 读取图像与几何证据，定位错误，调用受限工具，重新执行并验收结果。根本区别在于**被修复的变量**。GRASE 固定 DA3 输出的相机和深度，修复由这些输出搭建的房间外壳、开口、物体资产、物体变换与支撑关系；我们要修复 DA3、VGGT、π³-X、MapAnything 本身产生的相机、深度、对应关系与融合几何。前者是 compositional real-to-sim scene modeling，后者才是 upstream metric reconstruction recovery。

截至 2026-10-20，GRASE 的 arXiv v1 正文与官方源文件没有链接项目页或代码仓库。因此，下文关于实现的陈述只来自论文和 appendix，不把尚未公开的实现细节当作事实。[论文摘要](https://arxiv.org/abs/2610.21498) · [论文 HTML](https://arxiv.org/html/2610.21498) · [官方源文件](https://arxiv.org/e-print/2610.21498)

## 1. GRASE 实际在做什么

| 项目 | GRASE 的正式定义 |
|---|---|
| 输入 | 静态室内环境的单目视频。论文对真实视频均匀采 100 帧，对模拟场景采 60 帧。 |
| 固定前处理 | DA3 联合预测每帧 depth、confidence、intrinsics 和 camera-to-world pose；这些结果被反投影成 point cloud。SAM 3 提供 surface/object masks。[方法 §3](https://arxiv.org/html/2610.21498#S3) |
| VLM 中间状态 | surface/object inventory、跨视角实例对应、墙面邻接、opening-to-surface 关系、object support DAG。 |
| 输出 | 可编辑场景：显式房间表面与门窗开口、每个物体一个 mesh 和 world-space transform，以及物体与表面的 support relations。 |
| 优化变量 | 墙面/曲面参数与开口拓扑；物体 mesh 的选择、位置、朝向、尺度；局部 mesh part 的形状；接触与支撑关系。**相机与 DA3 observations 在墙面优化中固定，物体 registration 也明确固定 cameras。**[房间优化](https://arxiv.org/html/2610.21498#S3.SS1) · [物体注册 appendix](https://arxiv.org/html/2610.21498#A1.SS3) |
| Agent action | 选观察帧；决定 plane 或 parametric patches；构造或替换物体资产；调整物体 pose/scale；选择错误 mesh component 并局部拉伸/重对齐；重拟合 surface；修改材质；检查浮空、穿插、遮挡与 support。调用 frame extraction、DA3、image editing、TRELLIS.2 和 Blender MCP；数值工具执行 RANSAC/SVD、bounded least squares、ICP、silhouette coordinate search、距离计算等。[Agent appendix](https://arxiv.org/html/2610.21498#A1.SS6) |
| 闭环 | 从恢复出的相机渲染当前场景，与原视频比较；查看 missing/extra-region overlays、mask IoU、boundary distance、point-to-surface distance。每轮修改只在目标 discrepancy 改善且其他视角不退化时接受，否则 rollback；每个 stage 最多三轮。 |
| 评测对象 | **最终可编辑场景**的 architectural shell、object instances 和 assembled scene，不是上游重建器本身。 |
| Benchmark / baseline | 8 个 Replica 场景与 15 个 Imaginarium 场景；对比 SAM3D、GPT-6 vanilla、GPT-6 + TRELLIS.2 + point cloud。[实验 §4.1](https://arxiv.org/html/2610.21498#S4.SS1) |
| 指标 | object/scene/shell 的 Chamfer Distance、normal consistency、F-score@5 cm；object instance mIoU；输入视角 PSNR、SSIM、LPIPS。[指标表](https://arxiv.org/html/2610.21498#S4.T1) |

一个最直接的判别例子：如果 DA3 把一段相机轨迹估错，GRASE 仍使用这组相机来反投影、拟合和渲染，论文没有定义“修复相机”的 action；如果相机正确但沙发靠背比例错误，GRASE 会定位靠背，约束性拉伸局部 mesh，再做多视角验收。我们的任务正好应以前一种错误为目标，而不要把后一个任务再做一遍。

## 2. 最相近处与根本区别

| 维度 | GRASE | 我们应界定的任务 |
|---|---|---|
| 共同机制 | training-free；VLM + 数值/3D tools；图像和几何状态联合诊断；多步执行；执行结果验收；失败回滚 | 相同。这套 workflow 本身不能再当核心 novelty。 |
| 被诊断的 error | 墙/开口拓扑错误、资产形状错误、object pose/scale 错误、局部 part 比例错误、浮空/穿插/support 错误 | baseline-native camera pose、intrinsics、depth、cross-view correspondence、visibility/occlusion 与 fusion 错误 |
| Agent 改写的 state | Blender scene graph 和 compositional mesh representation | reconstruction state：camera graph/trajectory、pairwise matches、per-view depth/confidence、visibility 与 fused geometry |
| 上游模型角色 | DA3 是固定的 measurement producer；GRASE 不比较或修复其他 reconstruction backbones | DA3、VGGT、π³-X、MapAnything 是被审计和修复的对象；同一 recovery interface 必须跨模型成立 |
| 输出用途 | 编辑和仿真的语义化室内场景 | 更准确、覆盖更完整、仍为 baseline-native 的 metric reconstruction |
| 证据与评测 | Replica/Imaginarium；shell/object/scene mesh 与 rendering fidelity | DTU、ETH3D、ScanNet++、7Scenes、HiRoom 等标准重建 benchmark；camera AUC/ATE/RPE、depth/point-cloud accuracy-completeness/F-score、registered/valid views、compute |
| 成功判据 | 物体身份、形状、放置、空间关系和 input-view rendering 与观察一致 | GT 或官方 evaluator 证明 pose/depth/geometry 改善；语义上“看起来合理”不能替代 metric recovery |

因此，二者可以共享“诊断—工具调用—执行—验证—回滚”框架，但解决的是相邻的两层：**GRASE 修 downstream scene assembly；我们修 upstream multiview geometry。** GRASE 甚至能作为动机：它依赖固定 DA3 几何，所以 DA3 的错误会成为其不可修复的输入误差。但这只是问题接口上的推论，不应声称 GRASE 实验证明了该问题。

## 3. 已经不能再说的 novelty

下面这些 claim 已被 GRASE 直接覆盖，单独使用会站不住：

- “首个用 agent/VLM 闭环改进 3D reconstruction / real-to-sim 的方法”；
- “首个让 VLM 理解 3D 错误并调用工具修复”；
- “首个持续使用原始多视角图像，对 3D 中间结果反复验证”；
- “首个把错误局部化到 responsible component，再映射到 constrained action”；
- “首个在修改后做多视角验收、无改善则 rollback”；
- “首个 training-free 地组合 VLM、几何优化器、生成器和 Blender/MCP tools”；
- “首个以 DA3 几何作为 agentic 3D pipeline 的观测”。

“Agentic 3D reconstruction reasoning”也太宽，审稿人会合理地把 GRASE 视为同类。若继续使用这个总称，副标题和 task definition 必须立即限定为 **test-time recovery of upstream metric reconstruction**。

## 4. 不沦为 GRASE 复现的硬边界

1. **任务变量必须在 reconstruction backbone 内。** 至少能改变 pose、depth、correspondence/edge 或 fusion/visibility state；只修 mesh、平面、资产和支撑关系不算我们的任务。
2. **不要把 Blender authoring 当主工具。** 工具应是 matching/edge verification、pose graph or BA、depth consistency/calibration、visibility-aware filtering/fusion，以及对 baseline 的受控重执行。若核心工具变成 plane fitting、开门窗、TRELLIS 资产替换或 object support correction，就回到了 GRASE。
3. **必须跨 backbone。** 同一个诊断与 action abstraction 至少在 DA3、VGGT、π³-X、MapAnything 上评估。只拿 DA3 点云做后处理，会非常接近 GRASE 的前半段。
4. **主结果必须是上游 metric 指标。** camera 与 depth/geometry 的官方指标是 primary endpoint；rendering 和 semantic scene quality 只能辅助。
5. **baseline 必须包含非 Agent recovery。** 比较 untouched backbone、固定 recovery pipeline、同工具预算的 deterministic/search/oracle policy、one-shot VLM 与 multi-step agent，才能证明增量来自诊断和 sequential decision，而不是“工具写得更好”。
6. **错误集先由四个 backbone 的真实失败产生。** 先完成跨模型 failure mining，再定义可复用 action space；不要预设一套场景编辑工具后去找案例。

最稳妥的一句话任务定义是：

> Given the native intermediate states of a failed feed-forward multi-view reconstruction, a training-free agent diagnoses and repairs upstream camera, depth, correspondence, and fusion errors, and accepts actions only through standard metric reconstruction evaluators.

在这个定义下，GRASE 是最接近的 workflow precedent，但不是同一任务 baseline：它固定 DA3 camera/depth，生成并修正 semantic editable scenes；我们不生成资产或 scene graph，而是恢复被 GRASE 等下游系统依赖的 metric geometry。

## Primary sources

- Zhaorui Lin, Haohan Liu, Zhen Chen, Xiangru Huang. [GRASE: Geometry-native Real-to-sim Agentic Scene Modeling from Video](https://arxiv.org/abs/2610.21498), arXiv:2610.21498v1, 2026-10-15.
- [GRASE full HTML, including method, appendix, experiments, and limitations](https://arxiv.org/html/2610.21498).
- [GRASE official arXiv source package](https://arxiv.org/e-print/2610.21498). The v1 source package contains no project-page or code-repository URL.

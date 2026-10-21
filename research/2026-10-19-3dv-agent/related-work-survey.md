# [RW survey](https://docs.google.com/document/d/1X13Fi5gbiWErC4c5hkfoe4iqzudjUiAnuxgsgVudbok/edit?tab=t.0)：从重建失败到可执行修复

固定照片的 training-free 3D reconstruction reasoning｜2026-10-21

**TL;DR.** 四个 baseline、六个数据配置和 412 次运行表明，单纯换 backbone 只能解决一部分场景，不能成为 agent 贡献。当前最有依据的问题是：先用 support-aware 几何诊断定位异常，再让 VLM 结合 RGB 与局部几何状态判断失败来自 camera 还是 depth/fusion，并选择可执行修复。VLM 的增量尚未证明；如果它不能超过确定性的 support-aware selector，这条 agent 线就不成立。

## 1. 最近工作已经覆盖什么

直接重建模型 [VGGT（CVPR 2025）](https://github.com/facebookresearch/vggt)、[π³ / π³-X](https://github.com/yyfz/Pi3)、[DA3（ICLR 2026）](https://github.com/ByteDance-Seed/Depth-Anything-3)、[MapAnything（3DV 2026）](https://map-anything.github.io/) 和 [VGGT-Ω（CVPR 2026）](https://github.com/facebookresearch/vggt-omega) 从照片直接预测相机、深度或三维点。Ω 展示了重复纹理导致静止轨迹和重复建筑等失败，也说明更强 backbone 已能消除部分错误。

后处理方法已经不是空白：[GLUEMAP](https://arxiv.org/html/2605.26103v1#S4) 与 MP-SfM 优化相机；[GGPT](https://github.com/ChenYutongTHU/GGPT) 用稀疏几何约束修正稠密点图。它们证明固定照片的 reconstruction 可以被修复，因此 BA、GLUEMAP 和 GGPT 是必须比较的 repair baselines。

Agent 工作解决的是相邻但不同的任务：[AREA3D](https://github.com/TianlingXu/AREA3D) 主动获取新视角，[GRASE](https://arxiv.org/abs/2610.21498) 修复可编辑的仿真场景。我们的输入固定，目标是恢复真实多视角 reconstruction 的相机与几何。“检查—调用工具—重跑”本身已经不新。

## 2. 统一实测得到什么

我们在同一 103-scene manifest 上运行 DA3、VGGT、π³-X、MapAnything，共 412/412 次成功。数据为 DTU-49、DTU-64、ETH3D、ScanNet++、7Scenes、HiRoom；输入图像、评测、standard/visibility mining 和跨模型结果均通过独立审计。

DA3 在本次 aggregate metrics 上全部最佳：DTU-49 overall 1.734，DTU-64 AUC@30 0.994，ETH3D / HiRoom / ScanNet++ F-score 为 0.785 / 0.858 / 0.783。逐场景 oracle routing 仍有收益，但不稳定：7Scenes +0.049 F-score，HiRoom +0.061，ScanNet++ 仅 +0.008；ETH3D 与 DTU-64 更小。多跑几个模型再选结果只能作为高成本 deterministic baseline。

更关键的证据来自 ScanNet++ intervention。DA3 在 8 个“pose 已最好、geometry 仍输掉”的场景中，最大 conditional gain 全部来自 GT depth，而不是 GT camera。逐视图 cross-view residual 能定位高 depth error：20/20 场景相关方向一致，median Spearman ρ=0.404，BH q=8.58e-6；但 raw residual 会漏掉高 support、纹理复杂的坏视图。去除 support rank 的线性投影后，median ρ 从 0.363 升到 0.417，17/20 场景改善。该校准仍需在独立 development split 冻结。

## 3. 核心 gap

现有 direct models 一次性输出 reconstruction；现有 repair methods 优化预先指定的变量；现有 3D agents 主要改变视角或编辑可控场景。尚未被回答的是：**固定照片下，能否根据可验证的几何状态先判断哪一部分、哪一种变量失败，再在受限 action set 中选择有效修复？**

VLM 只有一个可能成立的增量：解释为什么数值 residual 在某个局部失真，并据此区分 camera repair、depth repair、fusion/filtering 或 no action。视觉案例提示 thin structures、occlusion boundaries、反射/透明表面和重复细节可能造成 residual 漏报，但目前只是待验证假设。

## 4. 能成立的实验

固定 source backbone 和 compute budget，先在独立 development split 冻结 action set 与 support-aware selector。最终比较 no action、BA/GLUEMAP/GGPT、model routing、raw residual selector、support-aware selector、single-shot VLM、multi-step agent 和 action oracle。主结果只看执行后的 pose/geometry gain、成功率与额外计算，不使用 LLM-as-judge。

论文能成立的条件不是 agent 会调用多少工具，而是：在同一工具与预算下，VLM/agent 能处理数值诊断无法区分的失败，并显著超过 support-aware deterministic baseline。否则应把工作降级为 reconstruction diagnosis benchmark，而不是声称 agentic recovery。

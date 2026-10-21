# 实验计划：Baseline 复现与失败分析

2026-10-20｜Baseline 与 failure mining 已完成

当前阶段不做 agent。先用统一协议复现现有 feed-forward 3D reconstruction，找出可重复失败、可观察诊断量和可执行修复空间，再决定 toolkit 与 workflow。

## 1. 数据与模型

同一 manifest 共 103 场景：DTU-49 22、DTU-64 13、ETH3D 11、ScanNet++ 20、7Scenes 7、HiRoom 30。DTU-49 评价 surface overall；DTU-64 评价 pose AUC@30；其余数据同时评价 pose AUC@30 与 geometry F-score。本次沿用 DA3 发布子集与 evaluator，是统一诊断 population，不等于完整官方榜单。

四个 baseline 为 DA3-Giant、VGGT、π³-X、MapAnything。全部使用官方预训练权重和原生预处理，只读 RGB；场景、图像清单与顺序固定，不添加 BA、修复或参数搜索。每次运行保存相机、逐视图 depth/point/confidence、融合前后几何、评测、cross-view residual/visibility/support、耗时、显存、配置和日志。

## 2. 完成情况

四个模型完成 412/412 次运行，零 execution failure；每个模型复核 5,282 张输入图，共 21,128 条 SHA-256 checks。standard 与 visibility mining 对每个模型均覆盖 103 场景、10,358 个有向视图对。四个 batch audit、analysis audit、跨模型 audit 与 final-mining audit 全部通过；远端与本地最终 CSV 逐字节一致。本地测试为 160 passed、4 skipped。

## 3. 主要结果

DA3 在本次 aggregate metrics 上全部最佳。7Scenes AUC@30 / F-score：0.867 / 0.495；DTU-49 overall：1.734；DTU-64 AUC@30：0.994；ETH3D、HiRoom、ScanNet++ F-score 分别为 0.785、0.858、0.783。

逐场景最优 backbone 相对最佳固定模型仍有 headroom：7Scenes +0.049 F-score、DTU-49 −0.350 overall、HiRoom +0.061 F-score、ScanNet++ +0.008 F-score；ETH3D 与 DTU-64 较小。因此 model routing 只能作为必须击败的 deterministic baseline，不能作为主要贡献。

## 4. Failure analysis

ScanNet++ 的 nested intervention 顺序为 native → GT extrinsics → +GT intrinsics → +GT depth。DA3 与 VGGT 的主要 F-score headroom 来自 depth；π³-X 在 extrinsics 与 depth 间分裂；MapAnything 三项均有空间。DA3 pose 最优但 geometry 输掉的 8 个场景中，最大 conditional gain 全部来自 depth。因此第一类 action 应是在固定 camera 后修 depth/fusion，而不是默认修 camera。

DA3 的 round-trip pixel 与 relative-depth residual 能预测 GT-extrinsics gain，却不能在场景级预测 GT-depth gain。逐视图 residual p95 能定位高 GT depth error：20/20 场景方向一致，median Spearman ρ=0.404，BH q=8.58e-6。但 raw residual 会漏掉高 support、纹理复杂的坏视图。去除 support rank 的场景内线性投影后，median ρ 从 0.363 提升到 0.417，17/20 场景改善（paired p=7.08e-4）。这只是探索结果，必须在独立 development split 冻结。

## 5. 下一阶段的唯一问题

在 support-aware numerical diagnostics 已定位局部异常后，VLM 能否结合 RGB 与几何状态判断 failure cause，并从固定 action set 选择真正改善 geometry 的 depth/fusion 操作？先冻结 raw residual 与 support-aware baseline，再比较 single-shot VLM、multi-step agent、no action、deterministic model routing 与 action oracle。主指标是 action 执行后的 F-score/overall gain、成功率和额外 compute，不使用 LLM-as-judge。

完整结果见 [final baseline and failure-mining report](final-baseline-failure-mining.md)。

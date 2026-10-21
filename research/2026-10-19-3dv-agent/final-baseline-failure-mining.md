# Final Baseline and Failure-Mining Report

2026-10-20

## 一个具体失败告诉我们什么

在 ScanNet++ 中，有 8 个场景出现同一种矛盾：DA3 的相机指标是四个模型中最好，但其 native geometry 不是最好。逐项替换真值后，这 8 个场景最大的 F-score conditional gain 全都来自 depth，而不是 extrinsics 或 intrinsics。也就是说，仅凭“重建不好”去修 camera 会选错变量；系统必须先区分 camera failure 与 depth/fusion failure。

更细的逐视图分析显示，cross-view residual 确实能定位 depth error，但它不是充分诊断。邻接视图 residual p95 与 GT depth error 在 20/20 个 ScanNet++ 场景中正相关，median Spearman ρ=0.404（BH q=8.58e-6）。同时，raw residual 最容易漏掉的坏视图往往拥有更高的 outgoing support（19/20 场景，median paired difference +0.248，q=4.01e-5）和 incoming support（18/20，+0.194，q=3.67e-4）。这给出一个可检验的 research question：数值诊断先定位候选区域，VLM 是否能利用图像内容解释 residual 为什么失真，并选对 depth/fusion action？

## 运行与审计

- Models: DA3-Giant、VGGT、π³-X、MapAnything。
- Population: 7Scenes 7、DTU-49 22、DTU-64 13、ETH3D 11、HiRoom 30、ScanNet++ 20，共 103 场景。
- 完成 412/412 model-scene runs，零 execution failure；四个模型分别复核 5,282 张输入图，共 21,128 条 hash checks。
- 每个模型的 failure mining 包含 103 scenes、10,358 directed pairs；standard 与 visibility population 一致。
- 独立远端 audit 全部通过；四组最终 data CSV 的本地重算与远端结果逐字节一致。
- 本地测试：160 passed，4 skipped；Pillow legacy mode 有 1 个弃用 warning。

## 统一协议下的主要结果

| Dataset / metric | DA3 | VGGT | π³-X | MapAnything |
| --- | ---: | ---: | ---: | ---: |
| 7Scenes AUC@30 ↑ | **0.867** | 0.847 | 0.855 | 0.826 |
| 7Scenes F-score ↑ | **0.495** | 0.475 | 0.440 | 0.465 |
| DTU-49 overall ↓ | **1.734** | 1.924 | 3.029 | 7.631 |
| DTU-64 AUC@30 ↑ | **0.994** | 0.853 | 0.954 | 0.820 |
| ETH3D AUC@30 ↑ | **0.911** | 0.818 | 0.875 | 0.826 |
| ETH3D F-score ↑ | **0.785** | 0.625 | 0.710 | 0.576 |
| HiRoom AUC@30 ↑ | **0.959** | 0.881 | 0.944 | 0.879 |
| HiRoom F-score ↑ | **0.858** | 0.553 | 0.780 | 0.393 |
| ScanNet++ AUC@30 ↑ | **0.983** | 0.944 | 0.957 | 0.897 |
| ScanNet++ F-score ↑ | **0.783** | 0.676 | 0.723 | 0.295 |

这些数字只代表本次固定 subset 和统一 evaluator，不能写成各 benchmark 官方 leaderboard 结果。

## 失败空间是否值得做

DA3 是所有列出的 aggregate metrics 上最强的固定模型，但逐场景最优 backbone 仍有 headroom：7Scenes 平均 +0.049 F-score，HiRoom +0.061，ScanNet++ +0.008；DTU-49 overall 平均可下降 0.350。ETH3D 与 DTU-64 的 routing headroom 很小。最难的 consensus scenes 包括 7Scenes/pumpkin、DTU/scan49、DTU-64/scan122、ETH3D/delivery_area、HiRoom/20241230/828824/cam_sampled_08、ScanNet++/1ada7a0617。

这同时限定了论文贡献：简单地调用多个 backbone 再选最优，只是高成本 model routing。后续方法必须在固定 source backbone 和明确 compute budget 下，击败 deterministic routing、raw residual、support-aware residual、single-shot VLM 与 no-action baseline。

## 当前结论与下一步

1. **先修什么已经有证据。** ScanNet++ 上优先研究固定 camera 的 depth/fusion repair；camera repair 仍保留给 residual 指向 pose 的场景。
2. **数值 baseline 不可跳过。** 一个 target-free 的探索性校准从 residual rank 中去掉 support rank 的线性投影后，median scene ρ 从 0.363 升至 0.417，17/20 场景改善（paired p=7.08e-4）。它必须在独立 development split 冻结。
3. **VLM 的增量尚未证明。** 当前视觉案例提示 thin structures、occlusion boundaries、reflective/transparent surfaces、repeated high-frequency geometry 与 clutter 可能解释 residual 的漏报；plain walls、oblique views 和低 overlap/support 可能产生误报。这些仍是假设，不能写成结论。
4. **下一阶段只回答一个问题。** 给定数值诊断定位的局部异常，VLM 是否能识别 failure cause，并从固定 action set 选择可验证地改善 geometry 的操作？评价以 action 执行后的 F-score/overall、成功率和额外 compute 为准。

Supporting reports: [GT factor audit](scannetpp-gt-geometry-factors-20.md) and [residual-disagreement review](scannetpp-da3-residual-disagreement-review.md).

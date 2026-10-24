# 主实验：运行入口与依赖

本仓库只运行重建 baseline、保存原始输出并评分。目前是 **6 种方法 × 87 个固定的 16-view 序列**：ETH3D 26、ScanNet++ 29、T&T（Tanks and Temples）32。87 是序列数，尚未确认对应 87 个独立场景。

精简后共 **73 个文件**，不是 73 个入口：7 个任务 launcher、10 个首次安装 launcher、47 个 Python 文件、4 份版本约束、3 个上游兼容补丁，以及 README 和 `.gitignore`。没有 Agent、修复工具或训练流程。

## 平时从哪里运行

| 阶段 | `launchers/` 下的入口 | 输入 → 输出 |
| --- | --- | --- |
| 1. 固定输入清单 | `recovery_main_baseline_prepare.slurm` | 已准备的 RGB 清单 → 各方法的 plan/release；校验视图顺序和文件身份 |
| 2. 运行六种方法 | **`main_inference.slurm`** | plan/release、权重、已验收参考 → 原始预测、相机、点图、耗时和日志 |
| 3a. 准备评分 GT | `recovery_main_baseline_gt.slurm` | 评测专用真值 → GT 点图档案；不向推理暴露 GT |
| 3b. 导出 π³ 原始点图 | `recovery_main_baseline_pi3_unmasked.slurm` | π³ 的 `BASELINE_BATCH` → 未掩膜点图，不重新推理 |
| 3c. 生成并核对档案 | `recovery_main_baseline_score.slurm` | `BASELINE_BATCH`、`PLAN_NAME`、`EXPECTED_METHOD`、`GT_PREPARATION` → 504 像素中间评分 |
| 4. 按数据集评分、汇总 | **`score_benchmarks.slurm`** | 既有 protocol-audit-inventory → 每序列评分和 `benchmark-summary.csv`；用 `DATASET` 选择三个数据集之一 |
| 独立的主动采集实验 | `area3d_run_scene.slurm` | 用 `SCENE` 选择 Replica 场景 → AREA3D 采集、PGSR 重建和渲染评分 |

已有完整原始输出、GT 和评分档案时，直接进入第 4 步。**504 是中间校验，最终表使用 `official518` 画布上的 `official_robust`。** 当前最终评分器会先检查旧档案可复现，因此 3a–3c 仍是复跑所需依赖。这里的 AUC@5/10cm 是逐序列几何指标，取值 0–1，越大越好；不等于 T&T 官方 F-score 排名。

`score_benchmarks.slurm` 每次完整评六种方法，按序列平均生成六行结果。漏跑、重复或协议不一致会报错，不会删掉失败序列再计算较好看的均值。改变批次后必须更新 scratch 上的评分 inventory，不能继续使用指向旧批次的清单。

## 推理入口怎样选方法

六种方法共用 launcher，但各自的模型执行器与环境保持独立。提交前都必须设置 `BASELINE_NAME` 和 `ACCEPTED_RUN`；后者是验收参考目录，用来检查模型配置、推理代码和输出约定，不能随便换成其他模型的结果。

| `BASELINE_NAME` | 实际执行器（`scripts/recovery/`） | GPU 与额外输入 |
| --- | --- | --- |
| `da3` | `run_da3_population.py` | B200；`ACCEPTED_RUN` 为验收批次根目录 |
| `vggt` | `run_native_population.py` | B200；VGGT 验收目录 |
| `pi3` | `run_native_population.py` | B200；原版 π³ 验收目录 |
| `mapanything` | `run_native_population.py` | B200；另需 `REPLAY_REFERENCE` |
| `amb3r` | `run_backend_population.py` | H100；另需 `RUNTIME_CHECK` |
| `ggpt` | `run_ggpt_population.py` | H100；另需 `RUNTIME_CHECK`、DA3 结果的 `PARENT_BATCH` |

所有路径都须指向已有、已验收的真实记录。launcher 不再默认选择某个历史作业号。独立方法可并行；GGPT 要等 DA3 完成。

设置好以上环境变量后，在 Oscar 提交：

```bash
# da3 / vggt / pi3 / mapanything
sbatch --gres=gpu:nvidia_b200:1 launchers/main_inference.slurm
# amb3r / ggpt
sbatch --gres=gpu:h100:1 launchers/main_inference.slurm
# 三个数据集分别提交，不使用 job array
sbatch --export=ALL,DATASET=eth3d launchers/score_benchmarks.slurm
sbatch --export=ALL,DATASET=scannetpp launchers/score_benchmarks.slurm
sbatch --export=ALL,DATASET=tandt launchers/score_benchmarks.slurm
```

## 其余文件为什么保留

- **模型执行与预处理**：`scripts/{da3,vggt,pi3x,mapanything,amb3r,ggpt}/` 和 `recovery/run_*`。不同模型的输入尺寸、坐标约定、原始输出不同，不能用同一套转换替代。`pi3x/preprocess.py` 和远端 `envs/pi3x` 是沿用的路径名；当前加载的是原版 π³ 权重，已去掉 π³X 权重下载。
- **共享几何与隔离**：`geometry_state.py`、`native_state.py`、`public_canvas.py`、`parent_population.py`，以及 `run_backend_population.py` 中的 `command_for`。负责 RGB-only 沙盒、点图坐标、输入身份检查，不是 Agent 工具。
- **输入与评分**：`prepare_main_*`、`stage_pinhole_gt.py`、`gt_canvas.py`、`export_pi3_unmasked.py`、`score_*`、`pointmap_score.py` 和 `scripts/evaluation/`。模型输出、GT 和评测画布之间的转换，以及缺失预测的分母处理在这里完成。
- **首次安装**：`launchers/setup/`。六个 baseline 各一份准备脚本；另有 AREA3D 的数据、VLM、运行环境、PGSR 四份。AMB3R 的资产和环境步骤、GGPT 的资产/环境/匹配器步骤、π³ 的环境和原版权重步骤已分别合并。不同 CUDA/PyTorch 版本仍保留独立环境。
- **AREA3D 专用支持**：`prepare_area3d_*`、`prepare_replica_habitat.py`、`run_area3d_audited.py`、`verify_area3d_replay.py`、`assemble_area3d_dataset.py`、`verify_pgsr_scores.py` 和三个补丁。它使用主动采集与 PGSR 公开默认配置，渲染指标单独报告，不混入固定图像的几何表。
- **磁盘管理**：只保留 `manage_storage.py` 一个入口。默认预览，显式 `--apply` 才执行已验证清理策略；保护原始结果和权威点图。

## 仓库之外还需要什么

唯一运行根目录是 `/oscar/scratch/zliu328/3dv-reconstruction`，代码放在 `code/runner`，Slurm 使用 `default` 账户。

现有上游源码、权重、环境、RGB/GT、验收参考、`configs/*_pinned.json` 和运行清单都在 scratch，**克隆这个仓库本身并不能从零跑实验**。`setup/` 用于恢复对应环境和资产；它也依赖已保存的版本清单。精确版本和必要校验字段保留在运行记录中，不提交 Git。有效结果及成本继续保留在 scratch。

这次精简合并了推理、评分和安装入口，删除了未调用的旧开发/汇总脚本，并合并了磁盘策略。模型推理、预处理和几何评分公式保持不变。

# Main Reconstruction Experiments

Scripts and Slurm launchers for six fixed-image baselines: DA3, VGGT, original π³, MapAnything, AMB3R, and DA3→GGPT. The saved input population contains 87 sequences of 16 views: ETH3D 26, ScanNet++ 29, and Tanks and Temples 32. These are not 87 verified independent scenes.

| Step | Entry points in `launchers/` |
| --- | --- |
| Restore environments and assets | Method-specific `*_prepare.slurm` files; `recovery_pi3_prepare.slurm` prepares original π³ |
| Prepare fixed inputs | `recovery_main_baseline_prepare.slurm` |
| Run inference | `recovery_main_baseline_{da3,native,amb3r,ggpt}.slurm`; the native launcher selects VGGT, π³, or MapAnything |
| Prepare GT and point exports | `recovery_main_baseline_{gt,pi3_unmasked}.slurm` |
| Score and validate archives | `recovery_main_baseline_score.slurm` produces the intermediate 504-pixel scores |
| Evaluate the corrected protocol | `audit_tandt_protocol_matrix.slurm` and `audit_remaining_benchmarks.slurm`; use `official_robust` on the `official518` canvas |
| Run AREA3D separately | `area3d_prepare_*.slurm`, `area3d_download_internvl.slurm`, then `area3d_run_scene.slurm` with `SCENE` set |

AREA3D uses active acquisition and public PGSR defaults. Its rendering metrics belong in a separate table from fixed-image geometry scores. This is not a claim of reproducing its published numbers or an official benchmark ranking.

Launchers target `/oscar/scratch/zliu328/3dv-reconstruction`, use the `default` account, and call code under `code/runner`. Read each launcher's required environment variables before submission. RGB-only inference and GT scoring remain separate.

Raw outputs, costs, exact runtime manifests, source revisions, and model weights stay on scratch, outside Git. These scripts depend on the existing manifests and prepared upstream repositories; they are not a standalone dataset distribution. Some completed-run environments and weights require restoration before rerunning. Runtime checksum checks remain enabled. Storage maintenance is in `scripts/recovery/manage_storage.py`.

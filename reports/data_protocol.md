# Gate A data protocol

## Released data and split

Gate A uses Ego4D v1 Object State Change Classification and Point-of-No-Return
Temporal Localization. The released files are:

- `fho_oscc-pnr_train.json`
- `fho_oscc-pnr_val.json`
- `fho_oscc-pnr_test_unannotated.json`

No examples are reassigned, sampled, relabeled, or generated. Validation is the
only labeled evaluation split available locally; test submission is separate.

## Published processing and evaluation

The primary implementation is the published CVPR 2024 `sapeirone/EgoPack`
codebase. It consumes Ego4D's released Omnivore Video Swin-L features (32-frame
window, 16-frame stride, 1536 dimensions), converts only annotation-referenced
feature files to memory-mapped NumPy arrays, and reports:

- OSCC: classification accuracy.
- PNR: mean absolute temporal localization error in seconds on positive clips.

The original official baseline and EgoVLP remain comparison implementations,
but their frame-extraction pipelines exceed the available storage budget.

## Storage rule

Canonical videos are out of scope: the annotation-derived full-scale request
was 784.0312 GB and the 540p train/validation request was 345.2962 GB before
frame extraction. Ego4D CLI 1.7.3 ignores UID filtering for non-video feature
datasets, so the complete official `omnivore_video_swinl` release is downloaded
(149.6726 GB). EgoPack still selects examples through the unchanged official
annotation splits. Its processed NumPy files are produced only for referenced
videos by EgoPack's own dataset class; no new loader or split is introduced.

## Sources

- Ego4D FHO benchmark: https://ego4d-data.org/docs/benchmarks/hands-and-objects/
- Ego4D canonical videos and clips: https://ego4d-data.org/docs/data/videos/
- Ego4D downloader CLI: https://github.com/facebookresearch/Ego4d/tree/main/ego4d/cli
- Official FHO baseline: https://github.com/EGO4D/hands-and-objects
- EgoVLP: https://github.com/showlab/EgoVLP
- EgoPack: https://github.com/sapeirone/EgoPack

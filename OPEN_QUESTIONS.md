# Open questions before the HPC pilot

1. Pin the official DINOv3 checkpoint/revision that fits the 6 × L40S budget.
2. Pin the official V-JEPA 2 ViT-L or ViT-g checkpoint/config and its released
   frozen-probe preprocessing/layer-output path.
3. Pin Qwen3-VL's revision and verify the official processor's handling of
   externally sampled frames. The same-frame/altered-temporal-metadata C4
   intervention remains blocked until this is understood from official code.
4. Export the official SSv2 train/validation label map and canonical C0
   timestamp schedule. The code rejects guessed FPS schedules.
5. Confirm the official NExT-QA prompt template. Native evaluation accepts
   only exact deterministic A--E/0--4 outputs and never synonym matches.
6. Run the 50--100 sample smoke test for each model and inspect at least 20
   C0/C1/C2/C3 visual sample sheets before the full pilot.
7. Do not choose the optional benchmark until B1/B2 produce stable evidence.

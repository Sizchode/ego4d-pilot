# Published upstream revisions

The experiment does not vendor upstream academic repositories. Run
`scripts/bootstrap_upstreams.sh` to fetch these exact commits under
`third_party/`; the directory is ignored by Git except for this file.

The authoritative machine-readable pins are in
`configs/protocols/pilot.yaml`. Current pins are:

- Hugging Face Transformers: `6ec0f83563f56335d4227354614459a01d76ba0d`
- lmms-eval: `7e71cb99ddf8fe85c29e5f256dca5200cbd6211d`
- TAT-VQA: `7f6275b995492363f6ea9bc7ca42886be3c44d1d`
- AoT-PsyPhyBENCH: `2d62680ed58c1d6d4849f2bf665bd9214d9d1277`
- V-JEPA 2: `11b731c65ae7c964c256960ec6e3acf34d4eb29b`
- VideoMAE V2: `29eab1e8a588d1b3ec0cdec7b03a86cca491b74b`

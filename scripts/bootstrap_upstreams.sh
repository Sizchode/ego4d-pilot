#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
third_party_root="$project_root/third_party"
mkdir -p "$third_party_root"

clone_at_revision() {
  local directory="$1"
  local repository="$2"
  local revision="$3"
  local destination="$third_party_root/$directory"
  if [[ ! -d "$destination/.git" ]]; then
    git clone --filter=blob:none "$repository" "$destination"
  fi
  git -C "$destination" fetch origin "$revision"
  git -C "$destination" checkout --detach "$revision"
  [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]]
}

clone_at_revision transformers https://github.com/huggingface/transformers.git 6ec0f83563f56335d4227354614459a01d76ba0d
clone_at_revision lmms-eval https://github.com/EvolvingLMMs-Lab/lmms-eval.git 7e71cb99ddf8fe85c29e5f256dca5200cbd6211d
clone_at_revision TAT-VQA https://github.com/HanPT831/TAT-VQA.git 7f6275b995492363f6ea9bc7ca42886be3c44d1d
clone_at_revision AoT-PsyPhyBENCH https://github.com/mattashiho233/AoT-PsyPhyBENCH.git 2d62680ed58c1d6d4849f2bf665bd9214d9d1277
clone_at_revision VJEPA2 https://github.com/facebookresearch/vjepa2.git 11b731c65ae7c964c256960ec6e3acf34d4eb29b
clone_at_revision VideoMAEv2 https://github.com/OpenGVLab/VideoMAEv2.git 29eab1e8a588d1b3ec0cdec7b03a86cca491b74b

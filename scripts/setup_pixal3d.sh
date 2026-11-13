#!/usr/bin/env bash
# Sets up the optional Pixal3D image-to-3D model (pipeline/stages/image_to_3d/pixal3d_stage.py):
# the lyonsno/pixal3d-mlx port in vendor/pixal3d-mlx, its own conda env, and ~44GB of weights.
# Safe to re-run; each step is skipped when already done.
#
# Two weight files ship as PyTorch pickles. They are converted to safetensors here with
# torch.load(weights_only=True), which refuses to run code, instead of letting the port load
# the pickles itself (moge-mlx would use weights_only=False). Needs the meshy_local env (torch).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="$ROOT/vendor/pixal3d-mlx"
COMMIT=abc9a2e                       # tested 2026-11-05
ENV=pixal3d
HUB="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== code"
if [ ! -d "$VENDOR/.git" ]; then
  git clone -q https://github.com/lyonsno/pixal3d-mlx.git "$VENDOR"
fi
git -C "$VENDOR" checkout -q "$COMMIT"

echo "== conda env '$ENV'"
if ! conda env list | grep -qE "^$ENV\s"; then
  conda create -y -q -n "$ENV" python=3.12 >/dev/null
fi
(cd "$VENDOR" && PYTHONNOUSERSITE=1 conda run -n "$ENV" pip install -q -e .)

echo "== Pixal3D weights (TencentARC/Pixal3D, ~43GB)"
hf download TencentARC/Pixal3D >/dev/null
# The port hard-codes an older snapshot; its single-view weights are identical (same LFS ids)
# in the current one, so point the old name at whatever was downloaded.
SNAPS="$HUB/models--TencentARC--Pixal3D/snapshots"
OLD=0b31f9160aa400719af409098bff7936a932f726
if [ ! -e "$SNAPS/$OLD" ]; then
  CURRENT="$(cat "$HUB/models--TencentARC--Pixal3D/refs/main")"
  ln -s "$CURRENT" "$SNAPS/$OLD"
fi
# It also loads TRELLIS-image-large and DINOv3 from the cache; TRELLIS.2's setup downloads both.

echo "== NAF upsampler"
NAF="$VENDOR/weights/naf_release.safetensors"
if [ ! -f "$NAF" ]; then
  mkdir -p "$VENDOR/weights"
  curl -sfL https://github.com/valeoai/NAF/releases/download/model/naf_release.pth -o "$TMP/naf.pth"
  conda run -n meshy_local python -I -c "
import sys, torch
from safetensors.torch import save_file
sd = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
save_file({k: v.contiguous() for k, v in sd.items()}, sys.argv[2])
" "$TMP/naf.pth" "$NAF"
fi

echo "== MoGe-2 camera model (Ruicheng/moge-2-vitl)"
MOGE="$HUB/models--Ruicheng--moge-2-vitl/snapshots/converted-safetensors/model.safetensors"
if [ ! -f "$MOGE" ]; then
  mkdir -p "$(dirname "$MOGE")"
  hf download Ruicheng/moge-2-vitl model.pt --local-dir "$TMP/moge" >/dev/null
  conda run -n meshy_local python -I -c "
import sys, torch
from safetensors.torch import save_file
ck = torch.load(sys.argv[1], map_location='cpu', weights_only=True)
sd = ck['model'] if 'model' in ck else ck
save_file({k: v.contiguous() for k, v in sd.items() if torch.is_tensor(v)}, sys.argv[2])
" "$TMP/moge/model.pt" "$MOGE"
fi

echo "Pixal3D is ready: --image-model pixal3d, or pick it in the web UI."

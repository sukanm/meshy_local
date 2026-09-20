# meshy_local

A local-first macOS app that replicates the core [Meshy.ai](https://www.meshy.ai) workflow —
generating 3D models from text prompts — but runs entirely on your own Mac, with no cloud
dependency or subscription.

**Pipeline: text prompt → 3D mesh → watertight STL → sliced and printed on a Bambu Lab P2S.**

This is a personal tool built for one user's local workflow (Apple Silicon Mac + Bambu Lab P2S),
not a general-purpose product. It's optimized for "works reliably on my machine" over generality
or polish for other users/printers.

## How it works

Four swappable pipeline stages (`pipeline/stages/`, one `Protocol`-typed interface per step, wired
by `pipeline/orchestrator.py`):

1. **Text → image**: [`mflux`](https://github.com/filipstrand/mflux)'s Z-Image Turbo, MLX-native,
   run as an isolated `uv tool install` (its dependencies conflict with stage 2's).
2. **Image → 3D mesh**: [TRELLIS.2](https://github.com/microsoft/TRELLIS) by default, via a
   vendored MLX port (`vendor/trellis2mlx/`), or [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d)
   (`vendor/stable-fast-3d/`) as an alternative (`--image-model sf3d`).
3. **Repair / print-readiness**: a tiered escalation (`pipeline/stages/repair/`) — trimesh baseline
   repair → PyMeshLab topology repair → adaptive-resolution voxel remeshing — that guarantees a
   watertight, manifold mesh sized for FDM printing, with wall-thickness heuristics reported
   (never silently auto-fixed).
4. **STL export**: re-validates topology the way a slicer actually sees it (STL has no shared-vertex
   indices) and repairs any residual defects before writing the final file.

## Requirements

- Apple Silicon Mac (developed/tested on an M4 Max). The pipeline is MLX-native and pinned to
  `sys_platform == 'darwin' and platform_machine == 'arm64'`.
- [`uv`](https://docs.astral.sh/uv/) for Python/dependency management.
- A [Hugging Face](https://huggingface.co/) account with access requested/approved for the gated
  repos used by the image-to-3D stage(s):
  - `microsoft/TRELLIS.2-4B`, `microsoft/TRELLIS-image-large`
  - `facebook/dinov3-vitl16-pretrain-lvd1689m` (needs a separate Meta approval)
  - `stabilityai/stable-fast-3d` (if using `--image-model sf3d`)

  Authenticate locally with `hf auth login` (from the `huggingface_hub` CLI) — this writes a token
  to `~/.cache/huggingface/`, outside the repo. Nothing in this project reads a token from
  environment variables or committed files.
- A Bambu Lab P2S (or any FDM printer within a 256×256×256mm build volume) if you want to go all
  the way to a physical print. Slicing itself is left to Bambu Studio / OrcaSlicer — this project's
  job stops at producing a clean STL.

## Setup

```bash
# Main project environment
uv sync

# Text-to-image stage (isolated — conflicts with stage 2's transformers pin)
uv tool install mflux

# Image-to-3D stage: TRELLIS.2 (vendored, own venv)
cd vendor/trellis2mlx && uv sync && cd ../..

# Image-to-3D stage: Stable Fast 3D (only needed if using --image-model sf3d)
# — installed as part of `uv sync` above via pyproject.toml's local path sources
```

## Usage

### CLI

```bash
uv run python -m pipeline.cli "a small dragon figurine" --scale 80
```

Key flags:

| Flag | Meaning |
|---|---|
| `--scale <mm>` | Target longest dimension, in millimeters (required) |
| `--seed <int>` | Seed for reproducible generation |
| `--image-model trellis2\|sf3d` | Image-to-3D backend (default: `trellis2`) |
| `--smoothing <n>` | Taubin smoothing passes in the voxel-remesh repair fallback |
| `--detail <n>` | Voxel grid resolution for the voxel-remesh repair fallback |
| `--candidates <n>` | TRELLIS.2 only: generate `n` candidates, keep the cleanest by topology-defect count |

Output (raw mesh, intermediate artifacts, final STL, repair report) is written under `output/<job_id>/`.

### Web UI

```bash
uv run uvicorn server.app:app --host 127.0.0.1 --port 8420
```

Then open `http://127.0.0.1:8420`. Enter a prompt, scale, and optional seed; an "Advanced options"
panel exposes the image-to-3D model choice, smoothing, and mesh-detail sliders. Generation runs in
a background thread — `POST /generate` returns a job ID immediately, and the page polls
`GET /status/{job_id}` — since a full run can take several minutes. Only one job runs at a time.

### Tests

```bash
uv run pytest
```

## Project layout

```
pipeline/            # Core pipeline: stages, orchestrator, config, CLI
  stages/
    text_to_image/    # mflux (Z-Image Turbo)
    image_to_3d/      # TRELLIS.2 / Stable Fast 3D
    repair/           # Watertight/manifold repair, wall-thickness checks
    export/           # STL export + topology re-validation
server/               # FastAPI backend + static HTML/JS frontend (<model-viewer> preview)
vendor/               # Vendored third-party code (TRELLIS.2 MLX port, Stable Fast 3D)
scripts/              # Standing debug/repro scripts, reusing existing job output
tests/                # pytest suite
research/             # Background research notes (competitor analysis, model surveys)
output/               # Generated job artifacts (gitignored; not checked in)
models/               # Local model weight cache (gitignored; not checked in)
```

## Status

The core text → STL pipeline runs end-to-end and has been validated on real generations. Not yet
done: a confirmed successful physical print on the P2S — the current focus is closing the gap
between the pipeline's own wall-thickness heuristics and what actually survives FDM printing on
thin, spiky organic subjects.

## Third-party code and licenses

`vendor/` contains vendored copies of two upstream projects, each under its own license:

- `vendor/trellis2mlx/` — MIT ([LICENSE](vendor/trellis2mlx/LICENSE))
- `vendor/stable-fast-3d/` — Stability AI Community License ([LICENSE.md](vendor/stable-fast-3d/LICENSE.md)) —
  free for research/non-commercial use and for commercial use under ~$1M annual revenue; see the
  license file for full terms before any commercial use.

Model weights themselves are downloaded on first run from Hugging Face under their own respective
licenses/gated-access terms, and are never committed to this repo.

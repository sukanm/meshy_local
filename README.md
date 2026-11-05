# meshy_local

A local-first macOS app that does the core [Meshy.ai](https://www.meshy.ai) workflow on your own
Mac, with no cloud service or subscription: you type what you want, and it gives you a 3D model
that's ready to print.

**Text prompt (or a reference photo) → 3D mesh → watertight STL → sliced Bambu Studio project → printed on a Bambu Lab P2S.**

It's a personal tool for one setup (an Apple Silicon Mac and a Bambu Lab P2S), not a general
product. It's tuned to work reliably on that machine rather than to be general or polished for
other people's printers.

- [What it does](#what-it-does)
- [Requirements](#requirements)
- [Setup](#setup)
- [Using the web UI](#using-the-web-ui)
- [Using the CLI](#using-the-cli)
- [Reference images and web search](#reference-images-and-web-search)
- [Printing the result](#printing-the-result)
- [What a job produces](#what-a-job-produces)
- [Configuration](#configuration)
- [HTTP API](#http-api)
- [Architecture](#architecture)
- [Design notes and lessons learned](#design-notes-and-lessons-learned)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [Project layout](#project-layout)
- [Status](#status)
- [Third-party code and licenses](#third-party-code-and-licenses)

## What it does

A job runs these stages in order. Each stage is a swappable `Protocol` in `pipeline/stages/`, and
`pipeline/orchestrator.py` wires them together:

| # | Stage | What it does | Tool |
|---|---|---|---|
| 1 | **Text → image** | Generates a concept image from the prompt. Skipped when you supply a [reference image](#reference-images-and-web-search). | [mflux](https://github.com/filipstrand/mflux) Z-Image Turbo (MLX) |
| 2 | **Image → 3D** | Reconstructs a textured mesh from the image, removing the background first. By default it makes 3 candidates and keeps the cleanest. | [TRELLIS.2](https://github.com/microsoft/TRELLIS) (vendored MLX port), or [Stable Fast 3D](https://github.com/Stability-AI/stable-fast-3d) |
| 3 | **Repair** | Makes the mesh watertight and manifold, scales it to your size, and stands it upright (Z-up). Measures wall thickness. | trimesh, PyMeshLab, scikit-image |
| 4 | **Print prep** | Thickens walls by a uniform outward offset where they're too thin to print, flattens the bottom (or adds a plinth), and places the model on the plate. | Blender (headless): OpenVDB and the Manifold boolean |
| 5 | **Export** | Writes the STL, then checks its topology again the way a slicer reads it. | trimesh |
| 6 | **Preview** | Renders the final STL from four angles and writes a GLB of the printable mesh for the browser viewer. | Blender (headless) |
| 7 | **Slice** | Slices for the P2S into `model.3mf`, a project with all print settings applied. Reports print time and filament, and counts small features that get no plastic. | Bambu Studio CLI |

Stages 4, 6 and 7 are best-effort. Without Blender or Bambu Studio the job still succeeds; a
warning replaces thickening and the base, and there are no previews or `.3mf`.

A typical 80mm figurine takes about 10–20 minutes on an M4 Max, mostly TRELLIS.2 (2–9 minutes per
candidate). The first run after downloading weights is slower.

## Requirements

- **An Apple Silicon Mac.** It was built and tested on an M4 Max with 128GB. Everything is
  MLX-native or MPS.
- **Disk space:** about 55GB for model weights. Z-Image Turbo is about 30GB; the three TRELLIS.2
  repos are about 21GB.
- **[conda](https://www.anaconda.com/)** for the main environment (`meshy_local`). The
  text-to-image and TRELLIS.2 stages run in their own uv-managed environments, because their
  pinned dependencies conflict with the main one.
- **A [Hugging Face](https://huggingface.co/) account** with access approved for these gated
  repos:
  - `microsoft/TRELLIS.2-4B` and `microsoft/TRELLIS-image-large`;
  - `facebook/dinov3-vitl16-pretrain-lvd1689m`, which needs a separate, manually reviewed Meta
    approval;
  - `stabilityai/stable-fast-3d`, only if you use `--image-model sf3d`.

  Log in with `hf auth login`. The token lives in `~/.cache/huggingface/`; nothing in this repo
  reads a token from files or environment variables.
- **[Blender](https://www.blender.org/) 5.2+** at `/Applications/Blender.app` (or set
  `MESHY_BLENDER`). Optional; used for print prep and previews. Its bundled Python supplies OpenVDB
  and the Manifold solver, and nothing is installed into it.
- **[Bambu Studio](https://bambulab.com/en/download/studio) 2.x** at
  `/Applications/BambuStudio.app` (or set `MESHY_BAMBU_STUDIO`). Optional; used for slicing. Its
  built-in P2S system profiles are read and never modified.
- **Internet access**, only for the first model download and for [web image search](#reference-images-and-web-search).
  Generation itself is fully offline.
- **A Bambu Lab P2S**, or any FDM printer with a 256×256×256mm build volume, to actually print.

## Setup

```bash
# 1. Main environment (conda)
conda env create -f environment.yml
conda env config vars set PYTHONNOUSERSITE=1 -n meshy_local   # keep ~/.local packages out
conda activate meshy_local
pip install --pre torch==2.15.0.dev20260915 torchvision==0.30.0.dev20260916 \
    --index-url https://download.pytorch.org/whl/nightly/cpu
pip install -r requirements.txt
pip install --no-build-isolation -e vendor/stable-fast-3d/texture_baker \
    -e vendor/stable-fast-3d/uv_unwrapper

# 2. Text-to-image (isolated: mflux needs transformers>=5, SF3D pins 4.42.3)
uv tool install mflux

# 3. Image-to-3D: TRELLIS.2 (vendored MLX port with its own venv)
cd vendor/trellis2mlx && uv sync && cd ../..

# 4. Log in to Hugging Face (weights download on first use)
hf auth login
```

`requirements.txt` pins the exact versions from the last known-good environment. `pyproject.toml`
and `uv.lock` are left over from before the move to conda and aren't used for installation.

The vendored TRELLIS.2 code loads two repos at fixed snapshot hashes (`af44b45f…` for
`TRELLIS.2-4B` and `25e0d31f…` for `TRELLIS-image-large`; see `vendor/trellis2mlx/generate.py`).
If you pre-download them, download those revisions, not `main`.

## Using the web UI

```bash
conda run -n meshy_local uvicorn server.app:app --host 127.0.0.1 --port 8420
```

Open <http://127.0.0.1:8420>.

1. **Describe the object** in the prompt, then set the target longest dimension (mm) and,
   optionally, a seed.
2. **Optional: choose a reference image** under *Reference image*. You can search the web, paste
   an image link, or upload a file. The 3D model is then built from that photo instead of a
   generated image. See [Reference images](#reference-images-and-web-search).
3. **Optional: open *Advanced options*:**
   - image-to-3D model (TRELLIS.2 or SF3D) and number of candidates;
   - repair smoothing and detail;
   - wall thickening: auto, off or a custom offset;
   - base: flatten, plinth or none.
4. **Press Generate.** The page polls every 5 seconds and shows the current stage. It also shows
   the image the model is being built from and, once reconstruction finishes, the raw 3D model.
5. **When the job is done** you get:
   - a 3D viewer of the **printable model**, with a toggle back to the raw coloured generation
     (which is *not* what prints);
   - the **Bambu Studio slice summary**: print time, filament, settings, and the unprinted-feature
     check;
   - the print-readiness report, covering watertightness, size, what repair did, and any warnings;
   - four rendered previews of the final STL;
   - **Download STL**, **Download Bambu project (.3mf)** and **Open in Bambu Studio**.

Reopen any job with `http://127.0.0.1:8420/?job=<job_id>`. Only one job runs at a time; a second
Generate while one is running gets an error.

## Using the CLI

```bash
conda activate meshy_local
python -m pipeline.cli "a small dragon figurine" --scale 80
python -m pipeline.cli "Godzilla" --scale 80 --image ~/Pictures/godzilla.jpg
python -m pipeline.cli "Godzilla" --scale 80 --image https://example.com/godzilla.png
```

| Flag | Meaning |
|---|---|
| `--scale <mm>` | Target longest dimension in mm (required). |
| `--image <path or URL>` | Build from this photo instead of generating a concept image; the prompt becomes the job's name. |
| `--seed <int>` | Seed for reproducible generation. |
| `--image-model trellis2\|sf3d` | Image-to-3D backend (default `trellis2`). |
| `--candidates <n>` | TRELLIS.2: make `n` reconstructions and keep the cleanest (default 3; 1 disables). |
| `--wall-offset <mm>` | Grow the surface outward so every wall is at least 2× this thick. `0` turns it off; the default is auto, 0.3mm when walls are too thin. |
| `--base flatten\|plinth\|none` | Flat cut on the bottom (default), a solid plinth, or nothing. |
| `--smoothing <n>` | Taubin passes in the voxel-remesh repair fallback (default adapts to resolution). |
| `--detail <n>` | Voxel resolution for that fallback (default adapts to print size, about 0.2mm voxels). |

It prints the STL path, the Bambu project path with time and filament estimates, and the full
print-readiness report.

## Reference images and web search

The image model only knows what it was trained on, so a specific product, character or landmark
often comes out wrong. Both image-to-3D models work from an image and remove its background
themselves, so a real photo can go straight into reconstruction instead.

**Three ways to supply one:**

| Method | How |
|---|---|
| **Web search** | Type a search (it defaults to the prompt), click a result. The server downloads the full-size image and falls back to the search thumbnail if the site blocks downloads. |
| **Image link** | Paste a direct image address. Use right-click → *Copy image address*, not the page URL. |
| **Upload** | PNG, JPEG, WebP, GIF, BMP or TIFF, up to 25MB. |

**What makes a good reference image:**

- one object, fully in frame, on a plain background;
- a three-quarter or front view;
- no text overlays or watermarks across the object;
- at least 256px on each side.

Images are stored upright (from EXIF), converted to PNG (which drops photo metadata such as
location), and capped at 1024px, since TRELLIS.2 works at 512px anyway.

**Search providers**, tried in order until one returns results:

| Provider | Used when | Notes |
|---|---|---|
| [Brave Search API](https://brave.com/search/api/) | `BRAVE_SEARCH_API_KEY` is set | Official API; needs a free account and key. |
| DuckDuckGo images | always (via the [`ddgs`](https://pypi.org/project/ddgs/) package) | No account needed and good results, but unofficial, so it can break. |
| [Openverse](https://openverse.org/) | fallback | Openly licensed images only, so fewer toys and characters. |

**Safety:**

- **Public addresses only.** The server only fetches from public internet addresses, checked on
  every redirect, so a pasted link can't reach this Mac's own services or your local network.
- **Copyright.** Images found on the web are usually copyrighted. Printing one for yourself is the
  intended use; check the rights before sharing or selling a model made from someone else's photo
  or design.

Staged images live in `output/_references/` and are deleted after 7 days. A job keeps its own copy
as `reference_image.png`.

## Printing the result

1. **Open the project.** Click **Open in Bambu Studio**, or open `output/<job_id>/model.3mf`. The
   project already has the settings below; just send it to the printer.
2. **Check the slice summary** before printing. "Small features the slicer prints no plastic for"
   counts cross-sections, such as a spike tip or claw, that the slicer leaves empty. Those parts
   print shorter or not at all. The page warns above 20%.

**The settings baked into `model.3mf`** (see `pipeline/config.py`):

| Setting | Value | Why |
|---|---|---|
| Printer, nozzle | Bambu Lab P2S, 0.4mm | the target printer |
| Process | 0.12mm High Quality @BBL P2S | fine layers for figurine detail |
| Filament | Bambu PLA Basic | |
| Walls | Arachne, `min_feature_size` 10%, `min_bead_width` 50% | Arachne varies line width, so thin spikes print. The lower minimums let it print narrower tips; Bambu's defaults are classic walls and 25% / 85%. |
| Supports | tree (auto) | arms, jaws and tails overhang |

**Measured on the 80mm Godzilla** (job `7b03f950890b`, fully sliced):

| Model / settings | Unprinted features | Print time | Filament |
|---|---|---|---|
| No thickening, default Arachne | 65 / 280 (23%) | 85 min | 10.0 g |
| 0.3mm thickening, default Arachne | 39 / 260 (15%) | 87 min | 11.0 g |
| **0.3mm thickening, fine-feature Arachne (current default)** | **12 / 260 (5%)** | **88 min** | **11.1 g** |
| 0.6mm thickening, default Arachne | 4 / 207 (2%) | 92 min | 12.0 g |
| 0.3mm thickening, 0.2mm nozzle | 8 / 257 (3%) | 175 min | 10.1 g |

**If too many features go unprinted:**

- **Print larger.** Every feature gets thicker in proportion to the size.
- **Use a custom wall offset of 0.4–0.6mm.** It's sturdier, but rounds the tips.
- **Swap to the 0.2mm nozzle.** It's about twice as slow, and needs the P2S 0.2 nozzle profiles in
  `config.py`.

Lowering the Arachne minimums further (5% / 35%) made no difference.

## What a job produces

Everything for a job is in `output/<job_id>/`:

| File | What it is |
|---|---|
| `job.json` | Job state, settings, the repair report, slice results and file names. |
| `concept_image.png` / `reference_image.png` | The image the 3D model was built from. |
| `raw_mesh.glb` | TRELLIS.2/SF3D's raw textured output. |
| `candidate_selection.json` | Topology-defect score of each TRELLIS.2 candidate. |
| `print_ready_mesh.ply` | After repair. |
| `prepared_mesh.ply` | After print prep (thickening and base). |
| `model.stl` | **The final printable mesh** (mm, Z-up, resting on Z=0). |
| `model.3mf` | **Sliced Bambu Studio project** with print settings. |
| `model_view.glb` | The final mesh as glTF for the browser viewer. |
| `preview_*.png` | Front, side, back and three-quarter renders of the final STL. |

## Configuration

**Settings in `pipeline/config.py`:**

| Setting | Default | Meaning |
|---|---|---|
| `IMAGE_GEN_STEPS`, `IMAGE_GEN_SIZE_PX` | 20, 512 | Z-Image Turbo steps and size. More than 512px is wasted, since TRELLIS.2 conditions at 512. |
| `IMAGE_GEN_PROMPT_SUFFIX` | "single object, centered, white background … thick-walled, sturdy, solid form" | Appended to every prompt; measurably reduces thin walls. |
| `TRELLIS2_NUM_CANDIDATES` | 3 | Best-of-N reconstruction. |
| `WALL_OFFSET_AUTO_MM` | 0.3 | Offset applied in auto mode when walls are too thin. |
| `VOXEL_FILL_SEAL_VOXELS` | 2 | Gap size the repair fill seals (see [design notes](#design-notes-and-lessons-learned)). |
| `BASE_MODE_DEFAULT` | `flatten` | `flatten`, `plinth` or `none`. |
| `BAMBU_MACHINE_PROFILE` / `BAMBU_PROCESS_PROFILE` / `BAMBU_FILAMENT_PROFILE` | P2S 0.4, 0.12mm HQ, PLA Basic | Bambu system profiles used for slicing. |
| `BAMBU_PROCESS_OVERRIDES` | Arachne, fine features, tree supports | Changes on top of the process profile. |
| `SLICE_CHECK_WARN_FRACTION` | 0.20 | Unprinted-feature share that triggers a warning. |
| `IMAGE_SEARCH_MAX_RESULTS` | 12 | Search results shown. |

**Environment variables:**

| Variable | Meaning |
|---|---|
| `MESHY_BLENDER` | Path to the Blender executable (default `/Applications/Blender.app/Contents/MacOS/Blender`). |
| `MESHY_BAMBU_STUDIO` | Path to the Bambu Studio executable (default `/Applications/BambuStudio.app/Contents/MacOS/BambuStudio`). |
| `BRAVE_SEARCH_API_KEY` | Use Brave's official image-search API first. |

## HTTP API

The web UI is a thin client over these routes (`server/app.py`). There's no pipeline logic in the
routes, so another frontend could use them unchanged.

| Method & path | Purpose |
|---|---|
| `POST /generate` | Start a job and return `{job_id}` at once. Takes `prompt`, `scale_mm` and optionally `seed`, `image_model`, `num_candidates`, `smoothing_iterations`, `voxel_resolution`, `wall_offset_mm` (`null` = auto), `base_mode` and `reference_image_id`. Returns 409 if a job is already running. |
| `GET /status/{job_id}` | The job's `job.json`. |
| `GET /output/{job_id}/{file}` | Any file in a job folder (also serves `/output/_references/<id>.png`). |
| `POST /open-in-bambu/{job_id}` | Open the job's `.3mf` (or STL) in Bambu Studio on this Mac. |
| `GET /image-search?q=` | Web image search: `{provider, results:[{title, image_url, thumbnail_url, page_url, width, height}]}`. |
| `POST /reference-image/from-url` | Fetch and stage an image: `{url, fallback_url?, title?, page_url?, provider?}` → `{ref_id, image_url, width, height}`. |
| `POST /reference-image` | Upload. The raw image bytes are the body; `X-Filename` is an optional URL-encoded name. Returns the same as above. |
| `GET /defaults` | Default values for the UI's controls. |

Page files are served with `Cache-Control: no-cache`, so a browser always picks up an updated UI.

## Architecture

- **Pipeline package (`pipeline/`).** Plain-dataclass types (`types.py`), one `Protocol` per stage
  (`stages/*/base.py`), an orchestrator that runs the stages and saves `job.json` after each one,
  and a CLI. It never imports from `server/`.
- **Isolated subprocess workers.** Stages with conflicting dependencies run as subprocesses in
  their own environments, and data crosses as files:
  - mflux is a `uv tool`;
  - TRELLIS.2 has its own uv venv in `vendor/trellis2mlx/`;
  - Blender runs scripts in its bundled Python 3.13 (`pipeline/blender/scripts/`, which must not
    import `pipeline`), exchanging `.npz` files;
  - Bambu Studio is driven through its CLI.
- **Server (`server/`).** FastAPI plus a static HTML/JS page using
  [`<model-viewer>`](https://modelviewer.dev/). A job runs in a background thread and the page
  polls. One `Pipeline` per image model is built on first use (SF3D loads its weights when
  constructed).
- **Best-effort stages.** Print prep, preview and slice report a problem as a warning or a
  `SliceResult.error` instead of failing a job whose STL already exists.

## Design notes and lessons learned

These are the decisions that shaped the pipeline, each tested on real output rather than assumed.

**Repair**

- **Tier-3 repair is almost always needed.** TRELLIS.2's raw meshes are never watertight; its own
  mesh extraction already leaves holes and non-manifold faces. Swapping its decimator
  (`--qem-simplify`) doesn't help. Repair escalates: trimesh, then PyMeshLab, then voxel
  remeshing.
- **The hollow-shell bug (fixed).** `VoxelGrid.fill()` only fills cavities that are *fully
  enclosed*, and a leaky surface never is. The result was a double skin 0.01–0.06mm thick around
  an empty body, which explains every "85–97% of the surface is too thin" figure in earlier work.
  It also very likely made the first physical Godzilla print collapse.

  The fix is a morphological closing (`_solid_fill`): seal small gaps, flood the outside, erode
  back. On the same Godzilla, volume went from 494 to 13,071mm³ and the spikes stayed sharp without
  any thickening.
- **The detail-recovery snap.** After voxel remeshing, every vertex is pulled back to the original
  surface, which restores teeth, claws and scale texture. The result is checked and repaired
  (`mesh_topology.weld_and_repair`, including a fix for holes that share a vertex). If repair
  can't close it, the stage falls back to the smoothed, unsnapped mesh.
- **Decimation.** PyMeshLab's quadric edge-collapse with `preservetopology=True` is used.
  `fast_simplification` broke clean meshes into hundreds of separate pieces.
- **Sliver-edge cleanup.** Decimation and the snap occasionally create absurdly long triangles,
  which showed up as a "starburst" of rods in Bambu Studio. Edges longer than 15× the median are
  removed and the holes refilled.
- **`process=False` on reloads.** trimesh's default vertex merging, applied when reloading an
  earlier stage's output, once reintroduced 1.6M non-manifold edges.

**Orientation**

- **Z-up.** glTF is Y-up and STL/slicers are Z-up. Converting at load time fixed STLs that used to
  lie on their side.

**Wall thickness**

- **Thickening is a true offset** (OpenVDB, ~1s). Two other approaches were rejected:
  - *CGAL alpha wrap*, which bridges gaps between nearby features, so its volume is about 2× what a
    true offset can add;
  - *Voxel dilation*, which inflates the whole model.
- **Auto thickening is 0.3mm**, down from 0.6mm. With solid interiors only real spike tips are
  thin, and 0.6mm visibly blunted them.

**Blender and Bambu Studio**

- **Blender booleans.** The MANIFOLD solver silently returns its input unchanged when it judges the
  mesh non-manifold, so `base_boolean.py` falls back to EXACT.
- **The Bambu Studio CLI doesn't resolve profile `inherits`.** Without flattening you silently get
  0.2mm layers and a 0g filament estimate, so the slice stage flattens each profile chain first.

**Quality levers that worked**

- **Best-of-N candidates.** Different seeds on the same image give near-identical shapes but very
  different cleanliness: one test was 87% vs 22% of the surface too thin.
- **A sturdiness suffix on the prompt** cut the too-thin surface from 83% to 69% in a test.

**Looked at and not adopted**

- **Hunyuan3D 2.1 on a Mac** is capped at 256px textures.
- **Multi-view conditioning.** Z-Image can't produce consistent views of one object.
- **FlexiCubes re-extraction** was 2–4 orders of magnitude slower and hung on real meshes.
- **Normal-map displacement.** TRELLIS.2 outputs no normal or height maps.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Web UI is missing new buttons or shows the raw model after an update | The browser has old page files cached. Hard-refresh once with Cmd+Shift+R; the server now sends `no-cache`. |
| `PermissionError … .locks/models--…` | The Hugging Face cache (or a symlink target) is owned by another macOS user. Fix its permissions. |
| TRELLIS.2 can't find weights after downloading | They're in a different snapshot folder. Download the revisions in [Setup](#setup), not `main`. |
| Report warns that thickening or the base was skipped | Blender is missing or a step failed validation. The job continues without it; check `MESHY_BLENDER`. |
| `Slicing failed: … not found` | Bambu Studio isn't installed where expected; set `MESHY_BAMBU_STUDIO`. |
| A job stuck on RUNNING after a crash | Restart the server. The real error is in the server log. |
| `uv run …` uses the wrong Python | The main environment is conda now: use `conda run -n meshy_local …`. |
| Image search fails | DuckDuckGo may be rate-limiting or have changed. Wait a minute, set `BRAVE_SEARCH_API_KEY`, or paste an image link instead. |
| "The image site refused the download" | Some sites block downloads; pick another result (the thumbnail fallback is lower resolution). |

## Development

```bash
conda run -n meshy_local python -m pytest
```

Blender- and Bambu Studio-backed tests (including a real slice) are skipped automatically if those
apps aren't installed. Network-dependent code is tested with mocks.

`scripts/` holds standing debug scripts that rerun stages on an existing job's `raw_mesh.glb`
instead of generating again (`rerun_repair_at_scale.py`, the thickening spikes, and so on).

## Project layout

```
pipeline/
  cli.py, orchestrator.py, config.py, types.py
  reference_images.py   # web image search, safe image fetch, upload staging
  stages/
    text_to_image/      # mflux (Z-Image Turbo)
    image_to_3d/        # TRELLIS.2 (best-of-N) / Stable Fast 3D
    repair/             # watertight/manifold repair, wall-thickness checks
    print_prep/         # wall offset, base, plate placement (Blender)
    export/             # STL export + topology re-validation
    preview/            # renders of the final STL (Blender)
    slice/              # Bambu Studio CLI: .3mf, estimates, unprinted-feature check
  blender/              # Blender runner + scripts that run inside Blender
server/                 # FastAPI app, schemas, static HTML/JS UI
vendor/                 # TRELLIS.2 MLX port, Stable Fast 3D
scripts/                # debug/repro scripts that reuse existing job output
tests/                  # pytest suite
research/               # research notes (competitors, model surveys, hardware)
output/                 # job folders (gitignored)
models/                 # local weight cache (gitignored)
```

## Status

- **Works end to end:** prompt or reference image → printable STL → sliced `.3mf`, through both
  the web UI and the CLI.
- **The one physical print so far failed** by layer separation. That was before the hollow-shell
  fix, the sliver fix and print prep. The current Godzilla (`7b03f950890b`) looks right in Bambu
  Studio but hasn't been printed yet.
- **The next milestone** is a successful physical print on the P2S.

## Third-party code and licenses

`vendor/` contains copies of two upstream projects under their own licenses:

- `vendor/trellis2mlx/`: MIT ([LICENSE](vendor/trellis2mlx/LICENSE)).
- `vendor/stable-fast-3d/`: Stability AI Community License ([LICENSE.md](vendor/stable-fast-3d/LICENSE.md)).
  It's free for research, non-commercial use, and commercial use under about $1M annual revenue;
  read the file before any commercial use.

Model weights are downloaded from Hugging Face under their own licenses and gated-access terms, and
are never committed. Images found by web search belong to their owners.

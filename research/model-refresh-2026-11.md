# Are the pipeline's models still the best? (research, 2026-11-05)

Question: almost a year after the current stack was chosen (Z-Image Turbo for
text→image, TRELLIS.2 for image→3D, rembg for background removal), is each still
the best option that runs **locally on this Mac** (Apple Silicon, **48 GiB unified
memory**), for **printable geometry**, under a license that allows personal use?

Short answer: **TRELLIS.2 is still the right default; Z-Image Turbo has been
overtaken on leaderboards but is still a reasonable choice; the background-removal
model is the one clear, cheap upgrade.** One new image→3D model (Pixal3D) is worth
a head-to-head test.

## Image → 3D

| Model | Open weights / license | Runs on this Mac? | Evidence on quality |
|---|---|---|---|
| **TRELLIS.2-4B** (Microsoft, Dec 2025) — *current* | yes, MIT | yes — our vendored MLX port, ~4–4.5 min per candidate at 1024 (measured 2026-11-05) | Still described as the best downloadable model with the cleanest license (no territory/revenue limits) in Sept 2026 roundups. No successor announced; GitHub shows no release after Dec 2025. |
| **Pixal3D** (Tsinghua + Tencent ARC, SIGGRAPH 2026; May 2026, multi-view added Sep 2026) | yes, MIT (repo 2.6k stars) | MLX port exists (lyonsno/pixal3d-mlx — same author as our TRELLIS.2 port; 6 stars). M4 Max 128 GB: 512 ≈ 10 min, 1024 ≈ 21 min at ~33 GB peak; README says 1024 needs 64 GB, so on 48 GB it may only fit 512 | Built on the TRELLIS.2 backbone with pixel-aligned back-projection, so the shape follows the input image closely. Its paper beats original TRELLIS, Hunyuan3D-2.1, TripoSG, Direct3D-S2 by a wide margin on geometry (Toys4K IoU 93.6 vs 83.3 for Hunyuan3D-2.1, 79.5 for TRELLIS; user-rated fidelity 4.91 vs ≤3.21) — **but it does not compare against TRELLIS.2.** Limitation it states itself: sensitive to imperfect segmentation boundaries. A vendor blog calls it "experimental", with occasional artifacts. |
| Hunyuan3D 2.1 (Tencent, Jun 2025) | yes, community license (excludes EU/UK/S. Korea) | MLX port dgrauet/Hunyuan3D-2.1-mlx: 28 stars, unclear license, last push Jul 2026 | Previously rejected (Mac texture cap, slower). Hunyuan 3.0/3.1 (2025–26) are hosted-only, no weights. |
| SAM 3D Objects (Meta, Nov 2025) | yes, SAM License | in appautomaton/mlx-spatial | Aimed at real-world photos with occlusion; no comparison against TRELLIS.2 found. |
| AssetGen (May 2026 paper) | no weights found | — | Claims TRELLIS.2 is less robust on complex inputs (lettering, fingers). |
| Tripo P1.0/H3.1, Meshy 7, Rodin Gen-2.5, Hunyuan 3.x | closed / hosted | no | Generally still better than any open model — not usable locally. |

The vendored TRELLIS.2 port (vendored 2026-09-17) has 20 upstream commits since then, mostly
CUDA-parity research; two touch mesh cleanup. Updating it is low priority.

## Text → image

Our use is narrow: a clean, centred, single-object "product photo" that conditions the 3D step.
Aesthetic leaderboards don't measure that, so any change should be judged on the resulting STL.

| Model | License | In mflux? | Notes |
|---|---|---|---|
| **Z-Image Turbo** (Nov 2025, 6B) — *current* | Apache 2.0 | yes | Was #1 open-weights on Artificial Analysis (Dec 2025). ~30 s per image here. |
| Qwen-Image-2.1 (20 Sep 2026, 7.1B + 8B text encoder) | Qwen Research License — non-commercial (fine for personal use) | yes (needs a newer mflux than installed) | Reported #1 open-weights on the Artificial Analysis arena in an Oct 2026 guide (couldn't confirm the live leaderboard). Has **native transparency** — could output the object already cut out. |
| FLUX.2 klein 4B / 9B (Jan 2026) | 4B Apache 2.0; 9B non-commercial | yes | Fast and small; edit mode. |
| Others added to mflux in 2026 (Krea 2, ERNIE-Image, Ideogram 4, FIBO 2, …) | various | yes | No evidence they're better for this use. |

Installed mflux is v0.19.2; latest is v0.24.0 (27 Oct 2026).

## Background removal — a concrete gap

The official TRELLIS.2 pipeline removes backgrounds with **BiRefNet** (`ZhengPeng7/BiRefNet`,
`trellis2/pipelines/rembg/BiRefNet.py`). Our MLX port calls `rembg.remove()` with its default,
**u2net** (2020), which is older and weaker. The silhouette directly shapes the 3D output, and
Pixal3D's paper says imperfect segmentation boundaries cause geometry artifacts.

The installed rembg (2.0.84) already offers `birefnet-general` (and `-lite`, `-massive`), so this
needs no new dependency. `generate.py` skips its own removal when the image already has
transparency, so the pipeline can cut the object out itself and pass an RGBA PNG.

## Recommendations, in order

1. **Background removal → BiRefNet** (rembg `birefnet-general`), done in our pipeline before
   TRELLIS.2. Cheap, matches the official pipeline. Verify on 2–3 subjects.
2. **Test Pixal3D against TRELLIS.2** on the same images, scored on the final STL — especially the
   reference-photo route, where matching the photo matters most. Check whether 1024 fits in
   48 GB; at 512 it may lose the detail advantage. Expect it to be 2–5× slower per candidate.
3. **Test Qwen-Image-2.1 against Z-Image Turbo** for concept images (upgrade mflux first), judged
   on the 3D result, not the picture.
4. **Keep TRELLIS.2 as the default** unless (2) shows a clear win.

## Sources

- TRELLIS.2: github.com/microsoft/TRELLIS.2 (BiRefNet in `trellis2/pipelines/rembg/BiRefNet.py`)
- 2026 release timeline: app.cinevva.com/guides/ai-3d-model-generation-timeline-2026
- Pixal3D paper: arxiv.org/abs/2605.10922; repo github.com/TencentARC/Pixal3D; MLX port github.com/lyonsno/pixal3d-mlx
- ComfyUI on Pixal3D vs TRELLIS.2: blog.comfy.org/p/trellis2-and-pixal3d-are-now-native
- 3D AI Studio comparison (vendor): 3daistudio.com/blog/pixal3d-vs-trellis-2-vs-hunyuan-3d-comparison
- AssetGen: arxiv.org/abs/2605.26137
- mlx-spatial: github.com/appautomaton/mlx-spatial
- mflux model list: github.com/filipstrand/mflux
- Qwen-Image-2.1: alternativeto.net/news/2026/9/alibaba-launches-qwen-image-2-1-…; orcarouter.ai/blog/qwen-image-2-1-open-weights-research-license
- Z-Image Turbo #1 (Dec 2025): x.com/ArtificialAnlys/status/2002839525609865575

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

## Result: Pixal3D vs TRELLIS.2 (tested 2026-11-05)

Setup: `vendor/pixal3d-mlx` at `abc9a2e`, conda env `pixal3d`, resolution 1024, seed 7, the same
BiRefNet cut-out for both models, then each raw mesh through the full pipeline (repair, auto print
prep, flatten base, export, Bambu slice) at 80mm. One seed per model per image (production TRELLIS.2
keeps the best of 3). Jobs are in `output/cmp-{trellis,pixal3d}-{godzilla,photo,dragon}`.

| | Godzilla concept | | Web photo | | Chibi dragon | |
|---|---|---|---|---|---|---|
| | TRELLIS.2 | Pixal3D | TRELLIS.2 | Pixal3D | TRELLIS.2 | Pixal3D |
| Generation time | 3.6 min | 7.3 min | 3.9 min | 8.3 min | 15 min* | 7.6 min |
| Raw boundary + non-manifold edges | 5,641 | 5,588 | 3,774 | 4,174 | 1,468 | 7,725 |
| Unprinted features (slicer) | 20/272 | 26/297 | 6/233 | 25/285 | 1/137 | 1/176 |
| Surface below 1.2mm (after print prep) | 16% | 11% | 11% | 10% | 1% | not measured |
| Matches the input's pose | no — leans forward, tail stretched out | **yes** | roughly | **yes** | yes | yes (face closer) |
| Unseen side (back/tail) | plausible | lumpy spikes | flat plank tail | **huge slab tail** | clean | **thin fin down the back, wavy tail plate** |

\* TRELLIS.2's dragon run was slowed by the Pixal3D install/download running at the same time.

Memory: Pixal3D peaked at 13.2GB GPU at 1024 on this 48GB Mac (the README's 33GB/64GB figures are
for an older version); ~7–8 min per run, about 2× TRELLIS.2.

Setup gotchas: the port hard-codes HF snapshot `0b31f91…`; the single-view weights are identical
in the current snapshot, so that folder is a symlink to it. MoGe-2 (`Ruicheng/moge-2-vitl`) ships
as a PyTorch pickle that moge-mlx would load with `weights_only=False`; it was converted once to
safetensors with `weights_only=True` instead. Without MoGe it silently uses a default 49° FOV
(MoGe estimated 23–24° on these images).

**Conclusion:** Pixal3D reproduces what the camera sees noticeably better — Godzilla's upright pose
and curled tail, the photo's walking pose — which is what its paper claims. But it invents worse
geometry for what the image doesn't show (a slab tail, a fin down the dragon's back), its surfaces
are bumpier, its raw meshes are no cleaner, and it's twice as slow. TRELLIS.2 stays the default.
Pixal3D could be worth offering as an option for reference photos where the pose matters, viewed
before printing.

## Result: Qwen-Image-2.1-Turbo vs Z-Image Turbo (tested 2026-11-05)

Setup: three prompts (Godzilla figure, owl figurine, small cute dragon figurine) with the usual
prompt suffix, seed 7. Z-Image Turbo at production settings (512px, 20 steps, 8-bit, mflux 0.19.2);
Qwen-Image-2.1-Turbo at its own defaults (1024px, 8 steps, 8-bit, mflux 0.24.0 in a separate conda
env `mflux024`). Each concept then went through the full production pipeline: TRELLIS.2 best of 3
(seeds 7–9), repair, auto print prep, export, Bambu slice, at 80mm. Jobs are in
`output/cmp-img-{zimage,qwen}-{godzilla,owl,dragon}`. One image per model per prompt.

| | Godzilla | | Owl | | Dragon | |
|---|---|---|---|---|---|---|
| | Z-Image | Qwen | Z-Image | Qwen | Z-Image | Qwen |
| Concept style | dramatic 3/4, big spikes | stocky, straight-on | white sculpted figurine | realistic painted owl | yellow chibi; a small loop on its head | green chibi with wings |
| Best candidate's raw defects | 17,453 | 26,603 | 125 | 5,533 | 1,348 | 146 |
| Unprinted features (slicer) | **16/243** | 46/204 | 0/87 | 0/81 | 22/244 | **1/122** |
| Surface below 1.2mm | 16% (0.3mm offset applied) | 13% | 0.3% | 1.7% | 16% | 4% |
| Seal widened (hollow-fill fix) | 3 | 3 | — | **6** | 3 | — |
| Visual result | spiky, characterful | cleaner, plainer | carved feathers and eyes | wavy surface, pits; painted plumage lost | **a tangled blob sticking out of the head** | clean: wings, eyes, spines |

Qwen took ~57s per image (vs ~42s), 23GB peak MLX memory, and 32.5GB of weights.

**Conclusion: no consistent winner; Z-Image Turbo stays.** Each model won one subject clearly
(Z-Image the Godzilla, Qwen the dragon) and Z-Image won the owl visually. The bigger factor is
*style*: Qwen tends toward realistic, painted objects whose detail is colour, which the STL can't
carry; Z-Image (with the "figurine… solid form" suffix) more often produces a sculpt whose detail is
shape. The Z-Image dragon's failure came from a small loop in the concept that TRELLIS.2 turned into
junk, not from the image quality as such. Not worth the non-commercial licence, the mflux upgrade,
and +32GB for this. A possible later test: a "single-colour unpainted sculpture" suffix for both.

Also found: the flatten base was skipped ("boolean result wasn't a closed manifold") on both Qwen
jobs and on the Pixal3D dragon, i.e. independent of the image model — a print-prep bug to fix.
Separately, TRELLIS.2 runs at ~47GB footprint on some subjects (owl, dragon) and swaps on this 48GB
Mac, ~14 min per candidate instead of ~4; that probably also explains the earlier
slow dragon run.

# DGX Spark / CUDA Hardware: What Actually Changes for messy_local

Research date: September 2026. Scope: if the user acquired one or two **NVIDIA DGX Spark** units (GB10 Grace Blackwell Superchip, 128GB unified LPDDR5x, NVLink-C2C, ConnectX-7 for pairing two units) instead of staying Mac/MLX-constrained, what actually gets better for the text→3D→STL pipeline? Companion to `local-text-to-3d-models-m4-max.md` (§5 in particular) and `meshy-ai-research.md` (§7) — this doc does not re-litigate the M4 Max survey, only what changes with CUDA + more memory available. Confirmed-vs-inferred flagged throughout, per this project's existing evidence bar. Not a printability-repair document — the Godzilla wall-thickness/thin-feature problem is a separate front and is only touched where a CUDA-only capability would plausibly help.

---

## 1. What DGX Spark's GB10 actually is, confirmed with real numbers

**Not a datacenter node.** GB10 is a single Blackwell-class GPU (sm_121 / compute capability 12.1) paired with a Grace CPU over unified memory, in a compact desktop form factor. It is explicitly a "personal AI supercomputer" / development box, not a rack-mount multi-GPU server.

**Confirmed spec numbers** ([flopper.io](https://flopper.io/compare/nvidia-dgx-spark-vs-nvidia-h100-sxm5-80gb)):
- 128GB unified LPDDR5x memory
- 273 GB/s memory bandwidth
- 140W TDP
- Marketed FP16 throughput: 125 TFLOPS (with sparsity/NVFP4 tricks baked into the marketing number, standard practice — see below for what's actually measured)

**Confirmed real-world measured throughput, not spec-sheet numbers** — two independent hands-on sources agree this is far below the marketed figure:
- [rossingram/Spark-DGX-Benchmark](https://github.com/rossingram/Spark-DGX-Benchmark) measured **~11–12 TFLOPs FP16/BF16** dense, against the same benchmark's own measured ~330 TFLOPs for an RTX 4090 and ~1000 TFLOPs for an H100 — i.e., DGX Spark achieved roughly **3.6% of a 4090's** and **1.1% of an H100's** measured throughput in this test, a far bigger gap than the naive "125 vs. 989 TFLOPS spec sheet ratio (~8x)" suggests.
- The [Level1Techs hands-on review](https://forum.level1techs.com/t/nvidias-dgx-spark-review-and-first-impressions/238661) independently reports the device sustains its rated FLOPS only in ~500ms bursts before hitting memory-bandwidth limits, and quotes an expert characterization that its real compute "sits basically like a [RTX] 5070" — a mid-range consumer card, not workstation/datacenter tier.

**Confirmed: it's memory-bound for LLM-style workloads, and memory-*capacity*-bound (not compute-bound) is where it actually wins** — demonstrated directly in §3 below (RTX 4090 OOM'ing on a workload DGX Spark completed).

**Verdict on the hardware itself, stated plainly:** DGX Spark's real value proposition here is **128GB of memory a single consumer/workstation GPU cannot match**, combined with **genuine CUDA + Blackwell driver/kernel support** — not raw speed. For a compute-bound generative model, expect DGX Spark to be *slower* than a high-end consumer GPU like an RTX 4090/5090 when the workload fits in the smaller card's VRAM, and to *only* pull ahead when the workload's memory footprint exceeds what a 24–32GB consumer card can hold at all.

---

## 2. Flash attention / Blackwell kernel support: the actual unlock for Hunyuan3D 2.1's texture cap

CLAUDE.md's rejection of Hunyuan3D 2.1 on Mac was specific: MPS has no FlashAttention-equivalent, capping the Mac fork at the "Safe" preset (256px textures) since full-resolution texturing's memory scaling assumes flash attention.

**Confirmed: DGX Spark does have working flash-attention-family kernels, with real but manageable ecosystem friction, not a hard blocker:**
- Flash Attention 3 does not support sm_121 (FA3's supported range tops out around sm_110); **Flash Attention 2 built for sm_120 is binary-compatible with sm_121** and has been used successfully ([vLLM GitHub issue #3170 discussion referenced via search](https://github.com/flashinfer-ai/flashinfer/issues/3170), [vllm-project/vllm#36821](https://github.com/vllm-project/vllm/issues/36821)).
- A real, working Hunyuan3D 2.1 Docker deployment on DGX Spark exists ([vij.app blog](https://vij.app/blog/posts/hunyuan3d-docker-nvidia-dgx-spark/)), building `custom_rasterizer` and `DifferentiableRenderer` CUDA components on-box, with confirmed VRAM figures: **10GB for shape generation, 21GB for texture generation, 29GB combined** — this is running the *full*, non-Safe pipeline (flash attention available, no MPS-style cap), comfortably inside DGX Spark's 128GB.
- The same author's separate benchmarking post ([vij.app/blog/posts/dgx-spark-benchmarks](https://vij.app/blog/posts/dgx-spark-benchmarks/)) gives **real, repeated-run timing**:

  | Task | DGX Spark | RTX 4090 (RunPod) |
  |---|---|---|
  | Hunyuan3D 2.1, shape only | ~40s (steady-state) | ~21s (steady-state) |
  | Hunyuan3D 2.1, shape + full texturing | ~185–220s (steady-state) | **CUDA out of memory — could not complete** |

  This is the single most concrete, decision-relevant data point found in this research: a 4090 is ~2x faster than DGX Spark when a workload fits in 24GB VRAM, but the *full-quality* Hunyuan3D 2.1 texturing pass (the exact thing that was capped to 256px on Mac for a different reason — no flash attention, not memory) **doesn't fit in a 4090's 24GB and OOMs**, while DGX Spark's 128GB completes it in under 4 minutes end-to-end. **Confirmed, not inferred**: this is a direct apples-to-apples run by the same author with the same workflow.

- Deployment reality check, also confirmed by direct testimony: the same author states **"I spent 8+ hours setting this up"** — building CUDA modules, Python 3.10, Blender's `bpy` API, and custom rasterizer kernels from source for sm_121, since prebuilt wheels for this very new architecture are inconsistent. A separate NVIDIA developer-forum thread on the same setup ([forums.developer.nvidia.com/t/dgx-spark-hunyuan3d-2-1](https://forums.developer.nvidia.com/t/dgx-spark-hunyuan3d-2-1/353211)) shows other users hitting build failures with `pillow-simd`, flash-attention compilation errors, and PTX errors before landing on a working CUDA 12.9.1 + custom-Docker recipe. **Net: this is a real, working path, not vaporware — but expect a comparable-or-worse first-setup tax to what CLAUDE.md already documents for the Mac MLX ports**, just for different reasons (bleeding-edge architecture vs. no-CUDA-at-all).

**Conclusion for §2: yes, DGX Spark genuinely removes the specific technical reason Hunyuan3D 2.1 was capped on Mac (no flash attention), and does so with confirmed real numbers, not marketing.** This is the clearest, most actionable finding of this whole research pass.

---

## 3. TRELLIS.2 on DGX Spark: full reference implementation confirmed running, no absolute timing found

**Confirmed: a working, Dockerized full-CUDA TRELLIS.2 deployment exists for DGX Spark** — [dr-vij/Trellis2-DGX-Spark-Docker](https://github.com/dr-vij/Trellis2-DGX-Spark-Docker) (19 stars, last pushed 2026-01-06, **no license file found in the repo metadata — check before relying on it**), building `nvdiffrast`, `CuMesh`, `FlexGEMM`, and `torchvision` from source for sm_121/CUDA 12.9. This is the **real Microsoft PyTorch/CUDA reference implementation**, not an MLX port — meaning no risk of the MLX port's acknowledged caveats (no-CUDA-parity-guarantee, topology failures on fine detail, texture smearing — see `local-text-to-3d-models-m4-max.md` §2) apply here at all, since it's the actual reference code.

**A real user report exists but with no absolute timing** ([note.com/npaka](https://note.com/npaka/n/n344aae232ac0?hl=en)): confirms successful end-to-end generation at up to 1536³ resolution (TRELLIS.2's own documented ceiling — same as the H100 benchmark numbers already in `local-text-to-3d-models-m4-max.md` §2), but the only quantified figure is setup time (~2 hours for CUDA + FlashAttention build + model download), not generation speed.

**One relative data point found, no absolute number**: an NVIDIA developer forum thread ([forums.developer.nvidia.com/t/trellis-2-3d-mode-gen-from-microsoft](https://forums.developer.nvidia.com/t/trellis-2-3d-mode-gen-from-microsoft/355030)) has a user (`propellerheadvij` — same author as the Hunyuan3D benchmarks above, going by username) stating **"It's almost 2x slower than Hunyuan 2.1, but I'm happy with the results."** Applying that ratio to the confirmed Hunyuan3D 2.1 shape+texture number (~185–220s) gives a **rough, unconfirmed estimate of ~6–8 minutes for full TRELLIS.2 on DGX Spark** — worth flagging clearly as *inferred from a casual forum comment*, not a measured number, but directionally useful: this would put DGX Spark's TRELLIS.2 runtime in roughly the same ballpark as `trellis2mlx`'s already-working **~8.6 minutes on M4 Max**, not a dramatic speedup — consistent with §1's finding that DGX Spark isn't a raw-speed win, it's a memory-and-correctness win (running the actual reference implementation, no MLX-port quality caveats, and headroom for larger models/candidate batches).

**Deployment friction, confirmed directly from the forum thread**: multiple other users besides the one who got it working reported build failures (`pillow-simd`, flash-attention compilation, PTX errors) before landing on a working recipe requiring CUDA 12.9.1 specifically. Same pattern as Hunyuan3D above — a working path exists, first-time setup is genuinely painful on this very new architecture.

---

## 4. Hi3DGen, TripoSG, Direct3D-S2 — re-checked, still CUDA-locked, still stale but not abandoned

Re-verified directly against live GitHub API data (not just re-reading the prior research pass):

| Model | Stars | Last push | License | Open issues |
|---|---|---|---|---|
| **Hi3DGen** (ByteDance) | 92 | 2025-09-16 | MIT | 3 |
| **TripoSG** (VAST-AI-Research) | 1,794 | 2025-04-18 | MIT | 46 |
| **Direct3D-S2** (DreamTechAI) | 1,282 | 2025-09-26 | MIT | 57 |

**All three have been dormant for roughly a year** — no code changes since the dates above, despite this research pass running a full year later (September 2026). This is a meaningfully different signal than "actively developed": these are **frozen research-release snapshots**, not projects gaining Blackwell/DGX-Spark-specific support or bugfixes. No DGX Spark-specific port, issue, or discussion was found for any of the three (searched specifically) — unlike TRELLIS.2 and Hunyuan3D 2.1, which both have real, working, documented DGX Spark deployments (§2, §3), **nobody has publicly done the equivalent porting work for Hi3DGen, TripoSG, or Direct3D-S2 on this hardware.**

This matters practically: getting any of these three running on DGX Spark would mean being the first to do that porting/dependency work (spconv, xformers, and similar CUDA-specific kernels rebuilt for sm_121), with no existing recipe to follow — a materially higher-effort, higher-risk undertaking than TRELLIS.2 or Hunyuan3D 2.1, where someone has already done and published the sm_121 build work. Given the stated VRAM requirements from the original survey (TripoSG: 8GB+; Direct3D-S2: 10GB@512³/24GB@1024³) all fit comfortably in DGX Spark's 128GB, so — unlike Hunyuan3D's flash-attention story — **the blocker for these three is porting effort, not hardware capability.**

No changed verdict on quality claims — nothing new was found beyond what `local-text-to-3d-models-m4-max.md` §5.2 already documented (Hi3DGen: best-in-class geometric precision per an independent survey; TripoSG: sharp features via SDF-VAE, lowest VRAM footprint; Direct3D-S2: 1024³ resolution via Spatial Sparse Attention). These remain **real capability gaps unreachable from Mac at all**, and DGX Spark would remove the *hardware* obstacle but not the *porting-work* obstacle.

---

## 5. Multi-view-consistent generation (the architectural gap from `meshy-ai-research.md` §7.1) — does DGX Spark help?

`meshy-ai-research.md` §7.1 already identified this as plausibly the single biggest architectural gap vs. competitors (Rodin, Zero123++-style pipelines) — not something any hardware change alone fixes, since it's a missing *pipeline stage* (a multi-view-consistent image generator sitting between Z-Image Turbo and TRELLIS.2), not a model that's currently too slow or capped.

What DGX Spark specifically changes here: §7.1 already found Zero123++ ships as a standard HuggingFace `diffusers` UNet pipeline with **no CUDA-exclusive kernels reported** — meaning it was already assessed as plausibly Mac-MPS-runnable, just unverified. On DGX Spark, this risk disappears entirely: a standard `diffusers` pipeline on real CUDA hardware is the best-supported, lowest-risk case in the entire ML ecosystem — no porting work of any kind would be needed, unlike every other item in this document. **This is the one candidate in this whole research area where DGX Spark isn't even necessary** — it was already the lowest-risk thing to try on the existing Mac, and remains so; DGX Spark just removes any remaining doubt about whether it "should" work.

---

## 6. Text-to-image front end: Z-Image Turbo, directly comparable numbers

Useful because this is the exact model the pipeline already uses (`mflux`'s Z-Image Turbo, MLX-native, `pipeline/stages/text_to_image/mflux_stage.py`), so this is a rare case of a true apples-to-apples comparison rather than a different model entirely.

**Confirmed, from the same benchmarking post as §2/§3** ([vij.app/blog/posts/dgx-spark-benchmarks](https://vij.app/blog/posts/dgx-spark-benchmarks/)): Z-Image Turbo on DGX Spark — 46.04s cold-start first run, then **~6 seconds per generation** steady-state.

Compare to CLAUDE.md's own measured M4 Max figure: ~30s for a 512px image at 9 steps (8-bit quantized MLX). The step counts/settings aren't confirmed identical between the two measurements (the DGX Spark figure's step count wasn't stated in the extracted text), so this isn't a fully controlled comparison, but the ~5x gap is directionally consistent with §1's finding that DGX Spark, while not H100-class, still meaningfully outpaces an M4 Max on compute-bound diffusion workloads specifically (as opposed to memory-bandwidth-bound LLM decoding, where the gap is smaller). Since this pipeline stage was already described in CLAUDE.md as "the easy, well-solved part," this isn't a meaningful lever either way — noted for completeness, not as a reason to prioritize anything.

---

## 7. Two DGX Sparks linked (ConnectX-7): does it matter here?

DGX Spark units can be paired via ConnectX-7 (200Gb/s) specifically to run models too large for one 128GB unit's memory. **None of the models relevant to this project's pipeline need this**: TRELLIS.2 (4B params), Hunyuan3D-DiT 2.1, Hi3DGen, TripoSG, and Direct3D-S2 all fit comfortably within a single GB10's 128GB with large margin (the largest confirmed figure found anywhere in this research is Hunyuan3D 2.1's 29GB combined shape+texture VRAM usage, §2). A second unit would only become relevant for this project if it later adopted something in Hunyuan3D 2.5's reported 10B-parameter shape-model tier (`meshy-ai-research.md` §7.2) — which remains hosted-only, not open-weight, as of this research pass — or for running multiple candidate generations in parallel (the existing best-of-N candidate-selection feature, `TRELLIS2_NUM_CANDIDATES`, could parallelize across two units instead of running sequentially). **Verdict: a second DGX Spark is not a meaningful lever for anything currently in scope; one unit's memory ceiling is not the binding constraint for any model surveyed.**

---

## 8. Net recommendation

**Highest-value swap, if this hardware became available**: replace the Mac/MPS-capped Hunyuan3D 2.1 fork with the full CUDA reference implementation via the confirmed working DGX Spark Docker recipe (§2). This is the one item in this document with hard, confirmed, favorable numbers on both axes that matter — **quality** (removes the 256px "Safe" texture cap entirely — full flash-attention-enabled texturing, confirmed running) and **speed** (~3–4 minutes full pipeline, confirmed, vs. ~14 minutes capped-quality on the Mac fork). It's also the one item where someone else has already absorbed the sm_121 porting pain and published a working recipe, so the remaining cost is real but bounded (a few hours of following a known-working Docker setup, not open-ended porting research).

**Second-highest value**: TRELLIS.2 on the real CUDA reference implementation via the also-confirmed-working Docker recipe (§3). Quality upside here is different in kind, not degree — it removes the MLX port's *acknowledged* caveats (no CUDA-parity guarantee, documented topology failures on fine detail, texture smearing on complex geometry) by running Microsoft's actual code, rather than raising a resolution cap the way Hunyuan3D's fix does. Speed upside is unclear and possibly minimal (§3's only data point is an inferred ~6–8 min estimate, in the same range as the current ~8.6 min MLX path) — **this is a correctness/quality-ceiling swap, not a speed swap**, and worth being honest about that distinction rather than assuming CUDA automatically means faster.

**What would NOT actually improve**: the printability/wall-thickness problem that caused the Godzilla print failure prompting this research thread. Nothing found in this pass suggests DGX Spark or any CUDA-only model changes the fundamental finding already documented in CLAUDE.md and `meshy-ai-research.md` §7.4 — that organic/thin-feature printability is an open, industry-wide problem (Meshy's own CEO describes it as unsolved on their end too), not something a bigger/better reconstruction model or a repair-stage swap fixes. A cleaner, less-messy raw mesh (which Hunyuan3D-full or TRELLIS.2-reference genuinely would provide, per §2–3) would reduce how much repair-stage work is needed and might modestly improve wall-thickness statistics the way best-of-N candidate selection already did (CLAUDE.md: 87.0%→22.2% below-floor swing between seeds on the *same* model) — but it does not represent a structurally different fix for organic-sculpt thin-wall printability. If the Godzilla failure investigation (the main conversation thread) concludes the fix needs to be architectural (e.g., adopting Tripo3D's prompt-steering trick more aggressively, or a genuinely different content strategy — printing mechanically simpler/thicker subjects), that conclusion holds regardless of what hardware generates the mesh.

**What's realistically NOT worth chasing even with this hardware**: Hi3DGen/TripoSG/Direct3D-S2 (§4) remain higher-effort than either Hunyuan3D-full or TRELLIS.2-reference, since nobody has done the sm_121 porting work for them yet and all three repos have been dormant for a year — this would mean being the first, with all three being narrower-scope (geometry-only, need pairing with a separate texture stage) research releases rather than complete pipelines. Not a dead end the way alpha-wrap or voxel-thickening were (CLAUDE.md) — just lower priority than the two items above, which already have working recipes.

---

## Sources

- [flopper.io — NVIDIA DGX Spark vs NVIDIA H100 SXM5 80GB](https://flopper.io/compare/nvidia-dgx-spark-vs-nvidia-h100-sxm5-80gb)
- [GitHub — rossingram/Spark-DGX-Benchmark](https://github.com/rossingram/Spark-DGX-Benchmark)
- [Level1Techs Forum — NVIDIA's DGX Spark Review and First Impressions](https://forum.level1techs.com/t/nvidias-dgx-spark-review-and-first-impressions/238661)
- [GitHub Issue — flashinfer-ai/flashinfer #3170, DGX Spark (SM121) Current Support Audit](https://github.com/flashinfer-ai/flashinfer/issues/3170)
- [GitHub Issue — vllm-project/vllm #36821, No sm_121 (Blackwell) support on aarch64](https://github.com/vllm-project/vllm/issues/36821)
- [GitHub Issue — vllm-project/vllm #31128, Add support of Blackwell SM121 (DGX Spark)](https://github.com/vllm-project/vllm/issues/31128)
- [vij.app — Hunyuan3D 2.1 in Docker on Nvidia DGX Spark](https://vij.app/blog/posts/hunyuan3d-docker-nvidia-dgx-spark/)
- [vij.app — DGX Spark — non synthetic tests (benchmark numbers)](https://vij.app/blog/posts/dgx-spark-benchmarks/)
- [NVIDIA Developer Forums — DGX Spark + Hunyuan3d 2.1](https://forums.developer.nvidia.com/t/dgx-spark-hunyuan3d-2-1/353211)
- [NVIDIA Developer Forums — Trellis 2 - 3d mode gen from microsoft](https://forums.developer.nvidia.com/t/trellis-2-3d-mode-gen-from-microsoft/355030)
- [note.com/npaka — Trying out 3D model generation with Trellis.2 on DGX Spark](https://note.com/npaka/n/n344aae232ac0?hl=en)
- [GitHub — dr-vij/Trellis2-DGX-Spark-Docker](https://github.com/dr-vij/Trellis2-DGX-Spark-Docker)
- [GitHub — bytedance/Hi3DGen (live repo metadata, checked this pass)](https://github.com/bytedance/Hi3DGen)
- [GitHub — VAST-AI-Research/TripoSG (live repo metadata, checked this pass)](https://github.com/VAST-AI-Research/TripoSG)
- [GitHub — DreamTechAI/Direct3D-S2 (live repo metadata, checked this pass)](https://github.com/DreamTechAI/Direct3D-S2)
- [NVIDIA — DGX Spark product page](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)

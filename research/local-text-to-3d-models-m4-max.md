# Local Text-to-3D Models on Apple M4 Max (48GB) — Research Notes

Research date: September 2026. Scope: given a Mac with an Apple **M4 Max** chip and **48GB unified memory**, what local (on-device, open-weight) options exist today for a **prompt → 3D mesh → STL** pipeline, and how well do they actually run on that hardware? This is a hardware-focused companion to the general Meshy.ai research (`meshy-ai-research.md`) and is meant to inform real build decisions for `messy_local`.

---

## 1. Hardware context: what the M4 Max / 48GB actually gives you

**Chip variants.** The M4 Max ships in two GPU configurations:
- 14-core CPU / **32-core GPU**, 410 GB/s memory bandwidth
- 16-core CPU / **40-core GPU**, **546 GB/s** memory bandwidth

A 48GB configuration can pair with either GPU variant depending on the specific Mac model/BTO options ([9to5Mac](https://9to5mac.com/2024/10/30/m4-max-chip-has-16-core-cpu-40-core-gpu-and-35-increase-in-memory-bandwidth/), [Apple](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/)). Either way, this puts the M4 Max well above the M2 Pro / M4 Pro machines most of the community ports below were validated on.

**Unified memory vs. discrete VRAM.** The single biggest practical advantage for generative-3D workloads: there is no hard VRAM ceiling separate from system RAM. A 24GB RTX 4090 has 24GB *dedicated* to the GPU and nothing more; the M4 Max's 48GB is shared between CPU and GPU with no PCIe copy overhead, so a model that "needs 24GB of VRAM" on paper can often just fit directly in unified memory without the multi-GPU or offloading tricks a discrete-GPU user would need. In practice, generative 3D models on Apple Silicon tend to peak at only a few GB of *active* memory during inference (see model-by-model notes below) — the 48GB ceiling is rarely the binding constraint; **compute throughput and missing CUDA kernels are the real bottlenecks**, not memory capacity.

**Software stack considerations:**
- **PyTorch MPS backend**: mature enough that most of these models run, but it is a second-class backend. Recurring pain points reported across projects: missing/slow fused-attention kernels (no FlashAttention-equivalent on MPS, so attention falls back to standard SDPA or chunked SDPA), operator gaps requiring `PYTORCH_ENABLE_MPS_FALLBACK=1` (silently falls back to CPU per-op, which is slow), and higher memory overhead than the CUDA backend for equivalent workloads.
- **CUDA-specific kernels**: several models (TRELLIS, Hunyuan3D's texture rasterizer, custom sparse-conv/hashmap ops) were written with hand-tuned CUDA kernels that simply don't exist on Metal. Every "Mac port" found in this research is a community fork that had to either (a) rewrite those kernels in pure PyTorch (slower, but portable) or (b) write native Metal/MLX replacements (faster, but requires the community to have done that specific work).
- **MLX**: Apple's own array framework is the most promising path for *good* Apple Silicon performance (native Metal kernels, no MPS translation tax), but 3D-generation coverage is early and community-driven rather than official. The one fully MLX-native pipeline found (TRELLIS.2 → `trellis2mlx`) is a solo/small-community project, not an Apple or Microsoft-official release.
- **GGUF/llama.cpp-style quantization**: not really applicable here. GGUF quantization is built around transformer LLM weight formats; 3D diffusion/flow-matching models use different architectures (sparse 3D convs, voxel/latent transformers) and the ports found use INT4/INT8 quantization via MLX's own quantization utilities rather than GGUF. One project (`trellis2mlx`) explicitly notes quantization reduced memory but **gave no measured speedup** — on Apple Silicon for this workload, compute is the bottleneck, not memory bandwidth to weights.
- **Text-to-image front end is mature**: unlike 3D generation, MLX-native Stable Diffusion and FLUX.1 (dev/schnell) implementations are well established (`mlx-examples/stable_diffusion`, `argmaxinc/DiffusionKit`), and run comfortably on Apple Silicon. This matters because, as covered below, essentially none of the good local 3D models take a text prompt directly — they take an image, so a local text-to-image step is a required pipeline stage, and this stage is the easy, well-solved part.

---

## 2. Survey of candidate models

### TripoSR (Stability AI / Tripo AI)
- **Type**: image-to-3D (not text-to-3D — despite marketing, it's a single feed-forward reconstruction model from one image; any "text-to-3D" framing bundles a separate text-to-image model in front of it) ([toolhunter.cc](https://www.toolhunter.cc/tools/trellis-mac), [tripo3d.ai blog](https://www.tripo3d.ai/blog/how-to-install-triposr)).
- **Requirements**: originally spec'd for a CUDA GPU with ≥6GB VRAM, ~8GB system RAM; runs on CPU as a fallback (slow).
- **Apple Silicon**: runs via PyTorch MPS backend; community reports of successful runs on M-series Macs (e.g., M2 MacBook Air, ~1 minute per object once set up via one-click installers like Pinokio).
- **Output**: mesh (OBJ/GLB), moderate quality, fast (~seconds to ~1 minute on capable hardware).
- **License**: MIT (code); permissive, good for local/personal use.
- **Verdict**: works, but its 2023-era output quality is now noticeably behind newer models (TRELLIS.2, Hunyuan3D 2.1, SF3D). Useful as a fast/cheap fallback, not the primary engine.

### InstantMesh
- **Type**: image-to-3D via sparse-view LRM (large reconstruction model) from a single image, using multi-view diffusion internally.
- **Apple Silicon**: no dedicated, actively-maintained Mac/MPS port surfaced in this research (unlike SF3D, TRELLIS.2, and Hunyuan3D, which all have specific community Mac forks). It's CUDA-oriented as shipped; likely runnable via generic MPS fallback with the same caveats as TripoSR, but no verified benchmark or fork was found.
- **Verdict**: not a first choice for Mac given the absence of a validated port — the effort of getting it running with unknown results is worse-spent than using SF3D or Hunyuan3D 2.1, which have people who've already done and documented that work.

### Stable Fast 3D (SF3D, Stability AI)
- **Type**: image-to-3D, fast feed-forward mesh reconstruction with UV-unwrapping and illumination disentanglement (separates baked lighting from albedo — notably better for print/paint use than models that bake shadows into the texture).
- **Requirements**: ~6GB for a single image on CUDA.
- **Apple Silicon**: **official, first-party MPS support** in the Stability-AI repo itself (not just a community fork) — uses custom Metal kernels for the texture baker. Requires `PYTORCH_ENABLE_MPS_FALLBACK=1` and the latest PyTorch (nightly recommended at time of writing). Stability's own docs recommend **32GB+ unified memory** for the MPS path, since MPS uses more memory than the equivalent CUDA run — the 48GB M4 Max clears this comfortably ([GitHub](https://github.com/Stability-AI/stable-fast-3d)).
- **Output**: GLB with baked PBR-ish textures, UV-unwrapped; configurable remeshing to triangle or quad topology with a target vertex count (non-binding) — this remeshing control is directly useful for producing print-friendly (lower, cleaner) topology rather than a raw, noisy reconstruction mesh.
- **License**: Stability AI Community License — free for commercial use up to **$1M annual revenue**; otherwise requires an enterprise license. Fine for a personal/hobbyist project.
- **Verdict**: **one of the two strongest candidates.** It is the only model in this survey with official (not community-hacked) Apple Silicon support, has explicit Mac hardware recommendations from the vendor, and its remeshing options are a direct asset for the print-readiness goal.

### Hunyuan3D 2.x (Tencent)
- **Type**: two-stage pipeline — Hunyuan3D-DiT (shape generation, diffusion transformer) + Hunyuan3D-Paint (texture generation) — from a single image or a few multi-view images.
- **Version note**: **2.1 (June 2025) is the last open-weight release.** 2.5, 3.0 (Sept 2025, reaches 1536³ resolution), and 3.1 (Nov 2025, 8-view input, watertight meshes, cleaner topology, 4K PBR) are hosted/API-only — Tencent has not open-sourced them. For a local pipeline, 2.1 is the ceiling today ([search summary](https://github.com/Tencent-Hunyuan)).
- **Apple Silicon**: multiple community forks exist:
  - `Brainkeys/Hunyuan3D-2.1-mac` — MPS acceleration, CUDA-free fallbacks; earlier reports suggested **untextured mesh output only** without a CUDA rasterizer.
  - `VladimirTalyzin/hunyuan3d-2.1-mac-rocm` — more complete: pure-PyTorch rasterizer (no compiler dependency), chunked SDPA attention for GPUs without flash-attention, Metal-specific fixes for `torch.isin` overflow and float64 conversion issues, Blender-free (`bpy`-free) mesh I/O. This fork claims **working PBR texturing on MPS**, verified end-to-end, though texture resolution is capped at 256px in the "Safe" preset (higher settings need flash-attention memory scaling that MPS lacks).
- **Benchmarks (from VladimirTalyzin fork, on Apple M4 Pro, 24GB, worst-case no-flash-attention)**:
  - Shape generation: ~5.7 minutes (7.6s/step)
  - Texturing: ~8.5 minutes
  - Note: on a 24GB Mac, texturing pushes into 8GB+ of swap. On a 48GB M4 Max this headroom problem should mostly disappear.
- **Output formats**: untextured geometry exports to **OBJ, GLB, PLY, STL, FBX, DAE, 3MF** — i.e., **direct STL export is already built in**, which is notable and directly relevant to this project. Textured output is GLB (PBR) or a Zip of OBJ+MTL+texture maps.
- **License**: two-layer — Tencent Hunyuan3D 2.1 Community License for the model/weights (note: **the license explicitly does not apply in the EU, UK, or South Korea** — check this if it matters for the user's jurisdiction), MIT for the Mac-port wrapper code.
- **Verdict**: **the other strongest candidate**, and arguably the best fit specifically for this project because it (a) has a genuinely-working community Mac port with documented PBR texturing, and (b) already exports STL natively, skipping a step other models require. Weaker points: pure community maintenance (no vendor support), slower than SF3D for texturing, and a license with a geographic carve-out to be aware of.

### TRELLIS / TRELLIS.2 (Microsoft)
- **Type**: image-to-3D via flow-matching transformers over a sparse-voxel 3D VAE ("O-Voxel" representation encoding geometry + appearance jointly). TRELLIS.2-4B is a 4B-parameter model.
- **Official requirements**: minimum **24GB NVIDIA GPU** (tested on A100/H100). Benchmarks on H100: ~3s at 512³, ~17s at 1024³, ~60s at 1536³ resolution ([HF model card](https://huggingface.co/microsoft/TRELLIS.2-4B)). This is a substantially higher-fidelity model than TripoSR/SF3D on paper (arbitrary topology, sharp features, full PBR with transparency/translucency support) — closer to what a hosted service like Meshy would actually be running.
- **License**: MIT for the model itself (DINOv3, used for image conditioning, has its own separate license terms via Meta — check before commercial use).
- **Apple Silicon**: this is where the most interesting recent work is:
  - `trellis2mlx` (lyonsno) — a **fully MLX-native, no-PyTorch, no-CUDA** end-to-end port. Native MLX DINOv3 conditioning, two-pass cascade architecture (low-res → upsample → high-res), Metal-accelerated QEM mesh simplification, xatlas UV unwrap. Minimum 16GB RAM (validated on M2 Pro); peak memory usage only **3–5GB during inference** (6.75GB peak RSS for a full run on M2 Pro 16GB). Output is textured GLB with PBR materials, configurable texture resolution (512² to 4096²) and face count (100K–200K typical).
    - **Benchmarks**: M2 Pro (16GB) full pipeline ≈ 21 minutes for 264K vertices / 200K faces. On **M4 Max (cascade mode)**, ≈ **8.6 minutes** full quality; an 8-step "recommended preview" mode runs in ~187 seconds, and a 4-step smoke-test in ~62 seconds.
    - **Caveats stated by the maintainer**: output quality is input-sensitive, no claim of exact parity with the CUDA/PyTorch reference implementation, localized topology failures around finely articulated details, texture smearing in complex areas (hair, horns, thin structures). INT4/INT8 quantization is supported but gives **no measured inference speedup** on this hardware — it only saves disk/memory footprint.
  - `pedronaugusto/trellis2-apple` and `deepikakoduri12/trellis2-apple` — separate community forks; `pedronaugusto`'s uses an MLX backend with custom Metal modules (`mtldiffrast`, `cumesh`, `flex_gemm`) specifically for mesh post-processing. Less benchmark detail surfaced than `trellis2mlx`.
  - A separate `trellis-mac` project (shivampkumar) instead uses the **PyTorch MPS route**: replaces CUDA-specific sparse 3D convolution with a gather-scatter equivalent, SDPA attention for sparse transformers, Python-based mesh extraction instead of CUDA hashmap ops. Reported at ~3.5 minutes for a ~400K-vertex mesh on an M4 Pro (24GB) — slower than the MLX-native route but a second independent validation that TRELLIS.2 is viable on Apple Silicon.
- **Verdict**: **the highest quality-ceiling option**, and the MLX-native port (`trellis2mlx`) is genuinely well-engineered — but it is a young, single/small-maintainer open-source project (not Microsoft- or Apple-official), so expect rough edges, breaking changes, and to potentially need to read/patch code yourself. Image-to-3D only, like SF3D and Hunyuan3D — needs a text-to-image front end.

### Shap-E / Point-E (OpenAI)
- **Type**: genuinely **native text-to-3D** (and image-to-3D) — Point-E outputs point clouds, Shap-E outputs implicit functions convertible to textured meshes or NeRFs. These are the only models in this survey that take a raw text prompt with no separate image-generation step.
- **Quality**: dated (2022–2023 era). Multiple sources characterize output as "decorative-quality at best" for anything beyond simple/blobby shapes — not suitable as a source of dimensionally-accurate, functional print geometry. Shap-E is the fastest of the pair (~3–5 minutes/model).
- **Apple Silicon**: runs via MPS/CPU without much friction since it's a smaller, older architecture; not the bottleneck here — quality is.
- **License**: MIT.
- **Verdict**: useful only as a low-effort baseline or fallback; not recommended as the primary engine given how far mesh quality has moved on since (SF3D, Hunyuan3D 2.1, TRELLIS.2 are all clearly ahead).

### Wonder3D / CRM / Zero123++ (multi-view diffusion approaches)
- These generate consistent multi-view images from a single input image/text-derived image, then reconstruct a mesh from those views (conceptually similar to what Hunyuan3D and SF3D do internally, but exposed as separate research pipelines).
- No dedicated, benchmarked Apple Silicon port was found for any of these in this research pass (unlike SF3D/Hunyuan3D/TRELLIS.2, which all have specific Mac work behind them). They would likely face the same CUDA-kernel and MPS-fallback issues as InstantMesh.
- **Verdict**: not recommended to pursue first — the effort-to-confidence ratio is worse than the three models above that already have validated Mac ports.

### Newer/other entrants (Rodin, CSM, LGM, Tripo's latest models)
- **Rodin, CSM**: commercial/hosted platforms, comparable to Meshy itself; no open local weights found.
- **Tripo (Tripo3D)**: remains a subscription SaaS; no open-weight local release surfaced for 2026 despite an active blog/marketing presence (including content specifically about *other* people's local models, which is a useful research signal but not a product they ship themselves).
- **Hunyuan3D 2.5 / 3.0 / 3.1**: as noted above, hosted-only, not open-sourced — 2.1 remains the open-weight ceiling from Tencent.
- No Apple-published local text-to-3D or image-to-3D generative model was found.
- **Verdict**: nothing new here changes the shortlist — the field's newest, best models (Tripo's latest, Hunyuan3D 3.x, Rodin) are all cloud-only, which is exactly the gap this project is trying to fill locally.

### MLX-native 3D generation projects specifically
Beyond `trellis2mlx` (covered above, the most complete example found), targeted searches for "MLX 3D generation," "MLX TripoSR," etc. did not surface additional mature, independent MLX-native 3D generation ports beyond the TRELLIS.2 family of forks and general MLX mesh-processing utility code. The TRELLIS.2/MLX work appears to be the current frontier of Apple-Silicon-native (not just MPS-fallback) 3D generation as of this research.

---

## 3. Practical recommendation

Ranked for **this exact hardware (M4 Max, 48GB)**, today:

**1. Stable Fast 3D (SF3D) — best first thing to actually try.**
Official vendor MPS support (not a third-party hack), explicit Mac hardware guidance from Stability AI itself, fast (~30s-class generation reported elsewhere for consumer GPUs; Mac will be slower via MPS but still workable), and built-in remeshing controls that directly help with print-readiness. Commercial-friendly license under $1M revenue. Start here for the fastest path to a working end-to-end pipeline.

**2. Hunyuan3D 2.1 (via the VladimirTalyzin Mac/ROCm fork) — best quality + native STL export.**
Slower (minutes, not seconds) and community-maintained rather than vendor-supported, texture resolution is capped on MPS (256px "Safe" preset), and there's a EU/UK/South Korea licensing carve-out to be aware of — but it already exports directly to STL, which removes a conversion step, and the fork's documentation shows real, specific engineering to make it work on Apple Silicon (not just "should work via fallback"). On 48GB of unified memory, the swap pressure that plagues the 24GB Macs in the benchmarks should be largely avoided.

**3. TRELLIS.2 via `trellis2mlx` — highest quality ceiling, but treat as experimental.**
The MLX-native architecture is the most technically elegant of the three (no PyTorch/CUDA translation tax, low peak memory at 3–5GB, genuinely native Metal performance), and the underlying TRELLIS.2 model is the newest/most capable architecture surveyed. But it's a young, small/solo open-source port with acknowledged topology failures on fine detail and texture smearing on complex geometry, no CUDA-parity guarantee, and 8.6 minutes per generation on M4 Max even in cascade mode. Worth building toward once the pipeline skeleton (from #1 or #2) is working, not the first thing to depend on.

**Honest caveat that applies to all three**: every one of these is **image-to-3D**, not text-to-3D. None of the currently-good local 3D models take a raw text prompt. The real local pipeline is:

```
text prompt → local text-to-image (MLX Stable Diffusion / FLUX.1, well-supported, fast, mature)
            → image-to-3D (SF3D, Hunyuan3D 2.1, or TRELLIS.2)
            → mesh cleanup/repair
            → STL
```

This two-hop pipeline is exactly how Meshy and most hosted "text-to-3D" services work internally too (see `meshy-ai-research.md`) — it is not a compromise unique to going local, it's the standard architecture for this problem today.

**A second, genuinely important caveat for the print-on-a-P2S goal specifically**: one source researched here ([localaimaster.com](https://localaimaster.com/blog/local-ai-3d-printing)) makes a pointed argument that at consumer/local hardware tiers, organic generative-mesh output (Point-E/Shap-E, and by extension the fancier models above) is **"decorative-quality at best"** for anything that needs to be a *functional* printed part — brackets, mounts, enclosures, anything with tolerances. Their recommended alternative for functional parts is a completely different local pipeline: have a code-capable LLM generate **OpenSCAD or CadQuery** (parametric CAD code) and compile that deterministically to STL, rather than generating a mesh directly. This is worth keeping in mind as a **complementary** approach, not a replacement: generative mesh models (SF3D/Hunyuan3D/TRELLIS.2) are the right tool for organic/decorative/artistic objects (figurines, props, decorative items — Meshy's actual bread-and-butter use case), while parametric-code generation is the right tool for functional, dimensionally-precise parts. The project's `CLAUDE.md` currently scopes to the Meshy-style organic-mesh workflow, which is the correct default given the stated inspiration — but if early real-world P2S prints of generated meshes come out looking "blobby" for anything with actual mechanical requirements, this is the documented reason why, and the fix is a different pipeline, not a better mesh model.

If none of the above worked out in practice, the honest fallback assessment would be: local text-to-3D mesh generation on Apple Silicon in 2026 is workable but meaningfully behind Meshy's hosted quality and speed (which almost certainly runs TRELLIS-class or larger models on multi-A100/H100 clusters) — expect **minutes, not seconds**, per generation, and expect to do real mesh cleanup afterward. That said, nothing found in this research suggests it's a dead end — SF3D and the Hunyuan3D 2.1 Mac fork in particular look like solid, buildable-on-today foundations.

---

## 4. STL export pipeline: mesh cleanup and repair

Whichever model produces the raw mesh (usually as GLB, since that's the common export format for SF3D and TRELLIS.2; Hunyuan3D's Mac fork can emit STL directly), the path to a **print-ready, watertight STL** in practice looks like:

1. **Load** the mesh with [`trimesh`](https://trimesh.org/) (Python), which reads GLB/GLTF, OBJ, PLY, STL, 3MF, and more, and is built with "watertight surfaces" as a first-class concern.
2. **Validate + basic repair**, using trimesh's built-in operations:
   ```python
   import trimesh
   m = trimesh.load("model.glb")
   m.process(validate=True)
   m.fill_holes()
   m.fix_normals()
   m.merge_vertices()
   print(m.is_watertight)
   ```
   This fixes the common cases: triangle winding order issues, inverted/inconsistent normals, small holes, and duplicate/nearly-duplicate vertices.
3. **Escalate to Manifold3D for harder cases**: `pip install manifold3d` — trimesh can call into Manifold3D (and/or Blender) for boolean operations and to fix non-manifold edges that trimesh's own heuristics miss. This is the step most likely to matter for AI-generated meshes specifically, since diffusion/reconstruction-based generation frequently produces small self-intersections, floating disconnected fragments, and non-manifold edges that simple hole-filling won't catch.
4. **Optional simplification/remeshing** for print practicality: SF3D exposes this natively (triangle/quad remeshing with a target vertex count) at generation time; for other models, `pymeshlab` or Blender's Python API (`bpy`) can do decimation and quad-remeshing as a post-process. Lower, cleaner topology also tends to slice and print more predictably than a raw, noisy reconstruction mesh.
5. **Export to STL**: `m.export("model.stl")` via trimesh once `m.is_watertight` is confirmed true.
6. **Downstream**: hand the STL to Bambu Studio / OrcaSlicer as planned in `CLAUDE.md` — this project's scope stops at producing a clean STL, not slicing.

For genuinely stubborn geometry, a documented technique is closing the loop with a vision-capable model: render the mesh in a slicer's auto-repair/preview mode, feed that screenshot to a vision-language model, and use its description of *where* the mesh is broken to target repair operations rather than blindly re-running generic repair heuristics.

---

## 5. Beyond the M4 Max: the unconstrained local quality ceiling

Follow-up research question: if memory/GPU constraints were lifted entirely (a proper Linux workstation or rented datacenter GPU — one or more A100/H100/H200-class cards, 24GB+ VRAM each, no MPS/Metal limitations), what's the actual open-weight quality ceiling in September 2026, separate from what's practically Mac-runnable today? This section is additive to §2 — it does not re-cover TripoSR/InstantMesh/SF3D/Hunyuan3D 2.1/TRELLIS.2/Shap-E/Point-E/Wonder3D/CRM/Zero123++, only what's new or was excluded from §2 specifically for having no Apple Silicon path.

### 5.1 Confirming the Hunyuan3D and TRELLIS ceilings haven't moved

- **Hunyuan3D 2.1 is still the open-weight ceiling from Tencent.** A GitHub issue opened July 17, 2025 on the official `Tencent-Hunyuan/Hunyuan3D-2.1` repo ("Inquiry: Open-Source Plans for Hunyuan3D-2.5 and Hunyuan3D-PolyGen," [#111](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/issues/111)) remains open with **zero response from Tencent** as of this research pass — 2.5/3.0/3.1 remain hosted/API-only, unchanged from the §2 finding.
- **TRELLIS.2-4B is confirmed image-to-3D only** at the base-model level ([HF model card](https://huggingface.co/microsoft/TRELLIS.2-4B)) — some secondary comparison articles loosely describe it as supporting "text-to-3D," but this refers to community wrapper Spaces that bolt a text-to-image step in front, not a native capability of the base model, consistent with §2's finding. Official requirements are unchanged: 24GB+ NVIDIA GPU (A100/H100-tested), up to 1536³ resolution, MIT license. No newer TRELLIS version was found beyond TRELLIS.2-4B.

### 5.2 New CUDA-locked models with no Mac path — the real capability gap

These were absent from §2 specifically because none has any Apple Silicon/MPS port, community or official — they represent a genuine capability gap versus what's Mac-runnable today, not just a speed difference:

- **Hi3DGen** (ByteDance, [github.com/bytedance/Hi3DGen](https://github.com/bytedance/Hi3DGen), MIT license) — a geometry-focused model that decouples shape from appearance via a normal-map intermediate representation (image → normal map → normal-regularized latent diffusion → geometry), aimed specifically at reproducing fine-grained geometric detail that reconstruction-style models (TripoSR/SF3D/InstantMesh) tend to smooth over. An independent 2026 survey ([3daistudio.com](https://www.3daistudio.com/state-of-ai-3d-generation-2026)) ranks it "best geometry"/"best-in-class geometric precision" among open models. Depends on `spconv` and `xformers`, both CUDA-specific with no Metal equivalent — no Mac port exists or was found. Geometry-only; would need pairing with a separate texture stage (e.g. Hunyuan3D-Paint) for a textured result.
- **TripoSG** ([github.com/VAST-AI-Research/TripoSG](https://github.com/VAST-AI-Research/TripoSG), MIT) — from VAST-AI-Research, the academic-research arm behind Tripo3D's SaaS (distinct from Tripo3D's closed hosted models). A large-scale rectified-flow transformer over an SDF-based VAE (hybrid SDF/surface-normal/eikonal loss supervision, trained on 2M curated image-SDF pairs), aimed at sharp geometric features and complex structures. Notably **only needs 8GB+ VRAM** for inference — the exclusion from the Mac-runnable shortlist in §2 is purely about being CUDA-locked (no MPS/Metal port found), not about being too heavy for consumer hardware. If a Mac port ever appears, this would be a strong, low-resource candidate.
- **Direct3D-S2** ([github.com/DreamTechAI/Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2), NeurIPS 2025, MIT license) — a sparse-volumetric diffusion transformer using a novel "Spatial Sparse Attention" mechanism, reaching 1024³ resolution while cutting training cost dramatically (8×A100 vs. the ~32 GPUs prior methods needed for even 256³). For inference specifically: **10GB VRAM at 512³, ~24GB at 1024³** (the vendor doesn't recommend 512³ as an end product — 1024³ is the intended quality tier). Geometry-only output (`.obj` export shown in docs); tested on Ubuntu 22.04, Windows possible via community workarounds documented in repo issues, no macOS/MPS mention found anywhere.
- **Academic vecset-diffusion lineage (context, not top picks today):** Michelangelo and CraftsMan3D ([HKUST-SAIL/CraftsMan3D](https://github.com/HKUST-SAIL/CraftsMan3D), MIT, weights on Hugging Face) represent an earlier "implicit vecset + neural SDF/occupancy field" architecture family that CLAY, Hunyuan3D, and TripoSG all build on conceptually. **OpenCLAY** ([CLAY-3D/OpenCLAY](https://github.com/CLAY-3D/OpenCLAY)) is a community-driven open reimplementation of Tencent's originally-closed CLAY paper. These are useful for understanding the research lineage but are generally superseded in output quality by Hunyuan3D 2.1/TRELLIS.2/Hi3DGen today — not recommended as first picks even with unconstrained hardware.
- **NVIDIA Meshtron** ([arXiv:2412.09548](https://arxiv.org/pdf/2412.09548)) — solves a different problem worth distinguishing: not image-to-3D reconstruction, but **artist-quality mesh topology generation** (autoregressive mesh-sequence generation reaching 64K faces at 1024-level coordinate resolution, an order of magnitude beyond prior mesh-topology generators). Published as an NVIDIA research paper only — **no official weights or code release found**; only an unofficial community PyTorch reimplementation exists ([GauravPatil8/Nvidia-Meshtron-Pytorch](https://github.com/GauravPatil8/Nvidia-Meshtron-Pytorch)). Relevant as a "what could eventually replace generic remeshing/decimation with something closer to hand-authored topology" direction, not as something usable today.
- **NVIDIA Edify 3D** — NVIDIA's other 3D generative effort, shipped only as a hosted NIM microservice (powering Shutterstock's and Getty Images' generative-3D features), never open-weight, and reportedly discontinued as an NVIDIA NIM preview as of June 2025. Not a local option in any form.

### 5.3 What's still closed/hosted-only at the frontier (context for the quality ceiling)

Per the same 2026 survey ([3daistudio.com](https://www.3daistudio.com/state-of-ai-3d-generation-2026)): **ByteDance Seed3D 2.0** (claimed 69–89.9% human-preference win rate vs. competitors — the current claimed quality leader overall), **Tripo 3.0** (Tripo3D's flagship hosted model; their developer platform added four hosted-only models in May 2026 — a flagship text+image-to-3D model, a low-poly game-asset specialist, an auto-rigger, and a preset motion library — none released as open weights), and **Rodin Gen-2** (Deemos/Hyper3D, positioned for film-grade hero assets) all remain closed/API-only, alongside Meshy itself. TripoSR remains Tripo's only genuinely open-weight release.

### 5.4 What lifting constraints actually buys

Two distinct effects, worth separating:

1. **For models already on the M4 Max shortlist (TRELLIS.2 in particular):** lifting hardware constraints mostly buys **speed and settings headroom**, not a different model — full-resolution TRELLIS.2 (1536³) on an H100 runs in ~60 seconds vs. `trellis2mlx`'s ~8.6 minutes on M4 Max, and removes the MLX port's caveats (no CUDA-parity guarantee, acknowledged topology failures on fine detail, texture smearing on complex geometry) since you'd run Microsoft's original reference implementation directly. Same underlying model, meaningfully better reliability and iteration speed.
2. **For CUDA-locked models with zero Mac path (Hi3DGen, TripoSG, Direct3D-S2):** lifting constraints buys **genuine additional capability** that isn't accessible on Apple Silicon at all today, regardless of patience — specifically, better fine-detail geometric fidelity (Hi3DGen, TripoSG) and higher raw resolution (Direct3D-S2's 1024³) than anything with a working Mac port currently achieves. This is the more honest "what are we giving up" answer: it's not that Mac-runnable models are crippled versions of the same models — it's that a few specific higher-geometric-fidelity models simply have no Mac path yet, community or official.

On whether any of this closes the gap to Meshy's own hosted quality (per `meshy-ai-research.md`): the honest answer is **partially, and the gap is narrower than the Mac-vs-full-hardware gap, not closed**. Even among *closed* frontier models (Seed3D 2.0, Tripo 3.0, Rodin Gen-2, Meshy 5/6 itself), the 2026 survey found "no single leader across all dimensions" — each is strongest in a different aspect (geometry, texture, speed). This suggests the best *open-weight* models with full hardware (Hunyuan3D 2.1 for texture/PBR, TRELLIS.2 or Hi3DGen for geometry) are already in the same competitive tier as hosted incumbents for specific dimensions, even if no single open model matches every dimension Meshy's stack covers simultaneously (Meshy's own advantage is more the *integrated pipeline* — remeshing, printability checks, rigging, community features — than raw single-model generation quality).

### 5.5 Practical path to this tier, if ever wanted

Not a deep cost analysis, just the realistic options: (a) **rent by the hour** on a cloud GPU marketplace (RunPod, Lambda Labs, Vast.ai, and similar — a single A100 40GB or H100 80GB instance is sufficient for every model surveyed here, including Direct3D-S2 at full 1024³) for occasional batch/high-quality runs without owning hardware; or (b) **a Linux workstation with a high-VRAM consumer GPU** (RTX 4090/5090-class, 24–32GB) — this alone covers TRELLIS.2 (24GB, at the edge), TripoSG (only needs 8GB), and Direct3D-S2 at 512–1024³ (10–24GB), i.e. most of this section's findings, without needing genuine datacenter hardware. Hi3DGen's exact VRAM figure wasn't published in the sources checked, but its dependency profile (spconv/xformers, no documented multi-GPU requirement) suggests it likely also fits a single high-VRAM consumer card. Worth revisiting this section if the user acquires either option later.

### 5.6 Licensing recap for newly-surfaced models

All newly-surfaced models in this section are **MIT-licensed** (Hi3DGen, TripoSG, Direct3D-S2, CraftsMan3D) — cleaner than Hunyuan3D 2.1's geo-restricted community license, no new licensing red flags found. NVIDIA Meshtron has no released weights/code to license (research paper only). OpenCLAY's license should be checked directly in its repo before relying on it (not verified in this pass, and CLAY's original paper predates any license commitment from Tencent).

---

## Sources

- [9to5Mac — M4 Max chip specs (16-core CPU, 40-core GPU, memory bandwidth)](https://9to5mac.com/2024/10/30/m4-max-chip-has-16-core-cpu-40-core-gpu-and-35-increase-in-memory-bandwidth/)
- [Apple Newsroom — Apple introduces M4 Pro and M4 Max](https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/)
- [willitrunai.com — MacBook Pro M4 Max 48GB: Best Local LLMs](https://willitrunai.com/macs/m4-max-48gb)
- [tripo3d.ai blog — How to Install TripoSR](https://www.tripo3d.ai/blog/how-to-install-triposr)
- [toolhunter.cc — TRELLIS Mac: Best Image-to-3D Tools for Apple Silicon](https://www.toolhunter.cc/tools/trellis-mac)
- [Hacker News — Show HN: Run TRELLIS.2 Image-to-3D generation natively on Apple Silicon](https://news.ycombinator.com/item?id=47828896)
- [GitHub — lyonsno/trellis2mlx (MLX-native TRELLIS.2 inference for Apple Silicon)](https://github.com/lyonsno/trellis2mlx)
- [GitHub — pedronaugusto/trellis2-apple](https://github.com/pedronaugusto/trellis2-apple)
- [GitHub — deepikakoduri12/trellis2-apple](https://github.com/deepikakoduri12/trellis2-apple)
- [Hugging Face — microsoft/TRELLIS.2-4B model card](https://huggingface.co/microsoft/TRELLIS.2-4B/blob/main/README.md)
- [GitHub — Brainkeys/Hunyuan3D-2.1-mac (README_macOS.md)](https://github.com/Brainkeys/Hunyuan3D-2.1-mac/blob/main/README_macOS.md)
- [GitHub — VladimirTalyzin/hunyuan3d-2.1-mac-rocm](https://github.com/VladimirTalyzin/hunyuan3d-2.1-mac-rocm)
- [Tencent Cloud Techpedia — Hunyuan3D on Apple Silicon (MLX Guide)](https://www.tencentcloud.com/techpedia/146529)
- [GitHub — Stability-AI/stable-fast-3d](https://github.com/Stability-AI/stable-fast-3d)
- [Hugging Face — stabilityai/stable-fast-3d](https://huggingface.co/stabilityai/stable-fast-3d)
- [Stability AI — Community License](https://stability.ai/license)
- [Meshy.ai — Meshy vs Hunyuan3D comparison (used for Hunyuan3D version history)](https://www.meshy.ai/compare/meshy-vs-hunyuan3d)
- [GitHub — ml-explore/mlx](https://github.com/ml-explore/mlx)
- [GitHub — ml-explore/mlx-examples (stable_diffusion)](https://github.com/ml-explore/mlx-examples/tree/main/stable_diffusion)
- [GitHub — argmaxinc/DiffusionKit](https://github.com/argmaxinc/DiffusionKit)
- [trimesh documentation](https://trimesh.org/)
- [localaimaster.com — Local AI for 3D Printing: Generate, Fix & Slice STL](https://localaimaster.com/blog/local-ai-3d-printing)
- [geekvibesnation.com — Best Local 3D Model AI 2026](https://geekvibesnation.com/best-local-3d-model-ai/)
- [biff.ai — Best Open-Source Text-to-3D Tools on GitHub (2026)](https://biff.ai/best-open-source-text-to-3d-tools-on-github/)

**§5 additional sources (unconstrained-hardware follow-up):**

- [GitHub Issue — Inquiry: Open-Source Plans for Hunyuan3D-2.5 and Hunyuan3D-PolyGen (#111)](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1/issues/111)
- [hunyuan3d.cc — Hunyuan3D Versions Explained: 2.0, 2.1, 2.5, 3.0 and 3.1](https://hunyuan3d.cc/hunyuan3d-versions)
- [Hugging Face — tencent/Hunyuan3D-2 model card](https://huggingface.co/tencent/Hunyuan3D-2)
- [Hugging Face — microsoft/TRELLIS.2-4B model card](https://huggingface.co/microsoft/TRELLIS.2-4B)
- [3daistudio.com — State of AI 3D Generation 2026: Market, Models, Open Source, MCP & APIs](https://www.3daistudio.com/state-of-ai-3d-generation-2026)
- [triposr.org — Hunyuan3D vs TRELLIS vs TripoSR (2026)](https://triposr.org/blog/hunyuan3d-vs-trellis)
- [GitHub — bytedance/Hi3DGen](https://github.com/bytedance/Hi3DGen)
- [Hi3DGen project page — stable-x.github.io](https://stable-x.github.io/Hi3DGen/)
- [arXiv — Hi3DGen: High-fidelity 3D Geometry Generation from Images via Normal Bridging (2503.22236)](https://huggingface.co/papers/2503.22236)
- [GitHub — VAST-AI-Research (org page)](https://github.com/VAST-AI-Research)
- [GitHub — VAST-AI-Research/TripoSG](https://github.com/VAST-AI-Research/TripoSG)
- [GitHub — DreamTechAI/Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2)
- [arXiv — Direct3D-S2: Gigascale 3D Generation Made Easy with Spatial Sparse Attention (2505.17412)](https://arxiv.org/html/2505.17412)
- [GitHub — HKUST-SAIL/CraftsMan3D](https://github.com/HKUST-SAIL/CraftsMan3D)
- [GitHub — CLAY-3D/OpenCLAY](https://github.com/CLAY-3D/OpenCLAY)
- [Hugging Face — CLAY paper page](https://huggingface.co/papers/2406.13897)
- [arXiv — Meshtron: High-Fidelity, Artist-Like 3D Mesh Generation at Scale (2412.09548)](https://arxiv.org/pdf/2412.09548)
- [GitHub — GauravPatil8/Nvidia-Meshtron-Pytorch (unofficial reimplementation)](https://github.com/GauravPatil8/Nvidia-Meshtron-Pytorch)
- [NVIDIA Developer Forums — High-Fidelity 3D Mesh Generation at Scale with Meshtron](https://forums.developer.nvidia.com/t/high-fidelity-3d-mesh-generation-at-scale-with-meshtron/316783)
- [Ultralytics — Simplifying 3D Model Creation with NVIDIA Edify 3D](https://www.ultralytics.com/blog/simplifying-3d-model-creation-with-nvidia-edify-3d)
- [Tripo3D — Introducing TripoSR](https://www.tripo3d.ai/research/triposr)
- [Tripo Developers — Models](https://developers.tripo3d.ai/en/models)

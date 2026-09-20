"""Central configuration: hardware/printer constants, thresholds, and which
concrete stage implementation is active for each pipeline step.

Kept as one importable module (not env-var soup) since this is a single-user,
single-machine project — see CLAUDE.md.
"""

from __future__ import annotations

from pathlib import Path

# --- Filesystem layout -------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
VENDOR_DIR = PROJECT_ROOT / "vendor"

# --- Printer constraints -------------------------------------------------
# Bambu Lab P2S build volume, confirmed via Bambu Lab's official spec page
# (https://bambulab.com/en/p2s/specs) on 2026-09-17. Do not guess this
# number for a different Bambu model — it varies (e.g. A1 mini is smaller).
P2S_BUILD_VOLUME_MM = (256.0, 256.0, 256.0)

# Practical FDM minimum wall thickness floor, per Meshy's own printing
# guide (research/meshy-ai-research.md, section 5.2). Treated as a
# report-only warning threshold, never an auto-fix trigger (see
# pipeline/stages/repair/trimesh_repair.py for the reasoning). Checked
# directly against real slicer guidance (2026-09-18 research round, not just
# assumed): for a 0.4mm nozzle, wall thickness that's an exact multiple of
# the nozzle diameter slices cleanly, and 1.2mm (3 perimeters) is the
# standard "fully robust" recommendation over a naive "1mm minimum" —
# 1.0mm is 2.5 extrusions, the worst case for gap-fill/over-extrusion
# artifacts. This floor is correctly calibrated, not overly conservative;
# don't relax it to reduce warning noise.
FDM_MIN_WALL_THICKNESS_MM = 1.2

# Assumed standard nozzle diameter for the P2S — the unit the wall-thickness
# severity tiers below are built from. If a different nozzle is ever used,
# this (and the two constants below, which are simple multiples of it)
# should move with it.
FDM_NOZZLE_DIAMETER_MM = 0.4

# Below this (1 nozzle width), a slicer can't form even one coherent
# perimeter — the wall is likely to come out as an actual gap/hole, not just
# a fragile-but-present feature. Distinguishing this from "merely below the
# robust floor" matters: a single near-zero-thickness point and a wall
# that's uniformly 0.9mm both used to get an identical "thin_walls" warning
# even though only the former is likely to be a genuine printing failure,
# not just a design tradeoff. Below FDM_MIN_WALL_THICKNESS_MM but at/above
# this is reported as "fragile" (1-2 perimeters, printable but not robust);
# below this is reported as "critical" (see trimesh_repair.py's tiered
# thin-wall message). Still never auto-fixed — see FDM_MIN_WALL_THICKNESS_MM
# above and trimesh_repair.py's module docstring for why.
WALL_THICKNESS_CRITICAL_MM = FDM_NOZZLE_DIAMETER_MM

# Fraction of sampled surface points allowed below the wall-thickness floor
# before the repair stage raises a WARNING (not BLOCKING) issue.
WALL_THICKNESS_WARNING_FRACTION = 0.05

# --- Repair stage tuning ---------------------------------------------------

# Below this fraction of total mesh volume, a disconnected shell is treated
# as reconstruction noise and auto-discarded (kept: the largest shell).
# Disclosed in the RepairReport regardless of size.
FLOATER_VOLUME_FRACTION_THRESHOLD = 0.02

# Number of surface points sampled for the wall-thickness heuristic.
WALL_THICKNESS_SAMPLE_COUNT = 2000

# Voxel grid resolution (voxels along the longest axis) for the tier-3
# marching-cubes repair fallback. Empirically tuned against a real, highly
# detailed TRELLIS.2 mesh (a Godzilla figure with fine dorsal spikes/claws):
# an old fixed default of 160 visibly destroyed that detail (spikes reduced
# to blocky stumps); ~350-400 preserves individual spikes/claws/teeth
# clearly, at a negligible extra cost (a few seconds).
#
# This is a *fallback* default only, used when the caller (the web UI's
# widget, or a CLI flag) doesn't explicitly request a resolution. The real
# default is geometry/scale-aware — see adaptive_voxel_resolution() below —
# because a *fixed* resolution in native mesh units silently gives worse
# real-world detail for larger prints: at resolution=350, an 80mm print
# gets a 0.23mm voxel pitch (comfortably finer than a ~0.4mm FDM nozzle),
# but a 256mm print of the exact same mesh gets 0.73mm — coarser than the
# nozzle, i.e. actively worse detail for a *bigger*, more detail-capable
# print. That's backwards, and was caught by computing the actual pitch at
# several target sizes rather than assuming a fixed resolution "just works".
VOXEL_REMESH_RESOLUTION = 350

# Desired real-world voxel pitch once the mesh is scaled to its target
# print size, independent of the mesh's native units or the requested
# print size. ~0.2mm is comfortably finer than a typical ~0.4mm FDM nozzle
# (so detail isn't the bottleneck) without being wastefully fine (which
# only costs more compute/memory for a print that can't resolve it
# anyway). Roughly matches VOXEL_REMESH_RESOLUTION's validated behavior at
# the 80mm scale most of this project's testing used (80/0.2 = 400,
# close to the empirically-tuned 350), while scaling correctly — finer
# absolute resolution for bigger prints, coarser for tiny ones — instead
# of holding resolution fixed in native units.
DETAIL_TARGET_VOXEL_PITCH_MM = 0.2

# Resolution is clamped to this range regardless of what the target pitch
# calculation suggests: below MIN, even a tiny keychain-scale print keeps
# basic shape fidelity; above MAX, a very large print (up to the P2S's
# 256mm build volume) is capped for practicality — voxel grid cost grows
# with resolution^3, and 500-600 was the top of what was performance-
# tested (a few seconds per stage even on Godzilla's ~570K-face raw
# marching-cubes output).
DETAIL_MIN_VOXEL_RESOLUTION = 250
DETAIL_MAX_VOXEL_RESOLUTION = 600


def adaptive_voxel_resolution(target_longest_dimension_mm: float) -> int:
    """Geometry/scale-aware default for VOXEL_REMESH_RESOLUTION: picks a
    resolution that gives a consistent real-world voxel pitch (see
    DETAIL_TARGET_VOXEL_PITCH_MM) regardless of the requested print size,
    clamped to a sane min/max for tiny or very large prints."""
    raw = round(target_longest_dimension_mm / DETAIL_TARGET_VOXEL_PITCH_MM)
    return max(DETAIL_MIN_VOXEL_RESOLUTION, min(DETAIL_MAX_VOXEL_RESOLUTION, raw))


# Taubin smoothing passes applied after voxel remeshing, to remove the
# staircase/blocky artifacts inherent to marching cubes over an axis-aligned
# grid. Validated on a real Godzilla mesh: visibly removes the faceting
# while keeping spikes/claws/teeth distinct, and only moves vertices so
# watertightness is unaffected. Used as a fallback fixed default (e.g. if
# a caller passes a resolution without a matching smoothing value); the
# real default is resolution-aware — see adaptive_smoothing_iterations.
VOXEL_REMESH_SMOOTHING_ITERATIONS = 10

# A finer voxel grid has proportionally smaller staircase steps to begin
# with, so it needs less smoothing to look clean — confirmed by rendering
# the same Godzilla mesh at resolution=600 with 0/3/10 smoothing passes:
# even 3 passes already looked clean, while fewer passes leaves more of
# the original marching-cubes surface (and thus more headroom for the
# detail-recovery snap afterward) intact. Linearly interpolated between the
# resolution clamp's two ends rather than a second fixed constant, so
# resolution and smoothing move together as one geometry-aware setting
# instead of needing to be tuned independently.
DETAIL_SMOOTHING_AT_MIN_RESOLUTION = 10
DETAIL_SMOOTHING_AT_MAX_RESOLUTION = 4


def adaptive_smoothing_iterations(resolution: int) -> int:
    """Resolution-aware default for VOXEL_REMESH_SMOOTHING_ITERATIONS."""
    span = DETAIL_MAX_VOXEL_RESOLUTION - DETAIL_MIN_VOXEL_RESOLUTION
    if span <= 0:
        return DETAIL_SMOOTHING_AT_MIN_RESOLUTION
    t = (resolution - DETAIL_MIN_VOXEL_RESOLUTION) / span
    t = max(0.0, min(1.0, t))
    value = DETAIL_SMOOTHING_AT_MIN_RESOLUTION + t * (
        DETAIL_SMOOTHING_AT_MAX_RESOLUTION - DETAIL_SMOOTHING_AT_MIN_RESOLUTION
    )
    return round(value)

# Minimum face count a reference mesh needs before the detail-recovery snap
# (see _voxel_remesh's docstring) is worth attempting. Real generative model
# output always has thousands+ of faces; this exists to skip the snap for
# trivially coarse references (e.g. a synthetic test cube), where snapping
# many voxel-remesh vertices onto a handful of reference triangles causes
# pathological vertex collapsing (many originally-distinct points landing on
# the same few target locations) rather than any real detail recovery —
# caught by a test using a deliberately-simplified synthetic fixture.
DETAIL_SNAP_MIN_REFERENCE_FACES = 100

# Marching cubes generates more triangles in regions with more surface
# curvature — a mesh with dense fine-scale texture (e.g. feather/scale
# bumps covering the whole surface) can produce a voxel-remesh output far
# larger than one with a smoother body and occasional sharp features, even
# at the same nominal VOXEL_REMESH_RESOLUTION. Caught during development: a
# feather-textured owl mesh produced 3.6M faces (vs. ~280K for an equally-
# detailed but smoother dragon figure at the same resolution), which made
# every subsequent per-vertex step (Taubin smoothing, the detail-recovery
# snap's nearest-surface query, and the export stage's weld-and-repair edge
# analysis) blow up to 10-30GB of RAM and minutes of runtime. If the raw
# marching-cubes output exceeds this many faces, it's simplified down to
# VOXEL_REMESH_SIMPLIFY_TARGET_FACES *before* smoothing/snapping — keeping
# every later step's cost bounded regardless of how bumpy the surface is.
# This is far more triangles than an FDM print's physical resolution can
# express anyway, so the simplification costs no real print-detail even
# though it looks aggressive on paper.
# Raised 2026-09-18 (400K/150K -> 700K/250K): these were tuned back when
# `fast_simplification` was the decimator and needed to be conservative to
# limit exposure to its bugs (see the fast_simplification incident above).
# Now that PyMeshLab's quadric edge-collapse is the decimator (fast and
# topology-safe on every real mesh tested), keeping more faces through
# smoothing/snapping is a close-to-free detail win worth the modest extra
# decimation time (a few extra seconds at these sizes).
VOXEL_REMESH_MAX_FACES_BEFORE_SIMPLIFY = 700_000
VOXEL_REMESH_SIMPLIFY_TARGET_FACES = 250_000

# Threshold (as a multiple of the mesh's own median edge length) for
# detecting degenerate "sliver" triangles introduced by PyMeshLab's
# decimation — see mesh_topology.find_sliver_edges for the full incident
# this was diagnosed from (a real user-reported Godzilla STL with long thin
# spikes shooting out from the body in Bambu Studio, traced to a decimation
# artifact present even before any smoothing or detail-recovery snap runs).
# A conservative middle ground, not tuned to the exact incident numbers
# (198-514 flagged edges were seen well above this threshold on the
# meshes that motivated it).
SLIVER_EDGE_LENGTH_MULTIPLE = 15.0

# (No fixed residual-defect threshold here: the detail-recovery snap's
# result is validated by actually attempting weld_and_repair on it — see
# _voxel_remesh — and falling back to the unsnapped mesh only if that
# repair itself can't fully close it, rather than rejecting based on a raw
# defect count. A junction-vertex-aware repair pass, added after finding
# defect counts alone were misleading, closes the vast majority of cases.)

# --- Detail-vs-printability optimization round (2026-09-18) ----------------
# Researched after the user asked for further detail-optimization ideas that
# still preserve printability. Two of four candidate ideas landed here;
# "raise the decimation ceilings" above is the third. The fourth idea
# (baking a normal/height map into real displacement geometry) was checked
# against the actual vendored TRELLIS.2 code and abandoned: its
# `texture_bake.bake_texture()` only produces `base_color` and
# `metallic_roughness` — there is no normal or height map anywhere in the
# pipeline to derive displacement from. Would need a different
# reconstruction model (or an entirely separate normal-estimation model) to
# even attempt; not pursued.

# Curvature-adaptive Taubin smoothing: rather than one uniform smoothing
# strength for the whole mesh, blend the fully-smoothed surface back toward
# the raw (pre-smoothing) marching-cubes positions in high-curvature regions
# (teeth, claws, fingertips) — directly targets the user's report that
# facial features and fingers were "too smooth" after uniform smoothing.
# Curvature is `mesh.vertex_defects` (discrete angle defect) on the
# pre-smoothing mesh: purely combinatorial from face angles around each
# vertex, needs no extra radius/scale parameter, and is near-zero on flat
# regions, large at sharp points/edges — exactly what's wanted here.
#
# 0.0 would mean "no smoothing at all at the highest-curvature vertices"
# (risks visibly staircased sharp points); 1.0 reproduces the old uniform
# behavior (fully smoothed everywhere, no adaptivity). 0.5 keeps some
# smoothing even at the sharpest features while meaningfully preserving
# them, and — as a side effect — leaves less for the detail-recovery snap to
# have to recover, making that step's job easier too.
CURVATURE_SMOOTHING_MIN_FACTOR = 0.5
# Percentile (not the raw max) used to normalize curvature into [0, 1]: a
# single extreme-curvature vertex (e.g. one claw tip) would otherwise
# saturate the whole normalization and make every other genuinely-detailed
# vertex look "flat" by comparison.
CURVATURE_SMOOTHING_PERCENTILE_CAP = 95.0

# Minimum-feature-thickness enforcement: locally thickens regions of the
# tier-3 voxel grid that have no local thickness anywhere near
# FDM_MIN_WALL_THICKNESS_MM (measured via a distance-transform local-
# thickness proxy — see _thicken_thin_voxels), instead of only reporting
# thin walls as a warning after the fact. Guarded against the detail-
# recovery snap undoing it: newly-thickened voxels are excluded from
# snapping, since pulling that new material back onto the original (thin)
# reference surface would silently defeat the whole point.
#
# **Disabled by default** — tested directly on Godzilla and found
# destructive, not just imperfect. The real issue isn't implementation
# detail, it's a mismatch between the technique and this kind of model:
# blanket morphological dilation assumes "thin" describes a few isolated
# protrusions on an otherwise-thick body, but on an organic/spiky sculpt
# where ~80-99% of the *entire* surface already reads as thin relative to a
# 1.2mm floor (see the wall-thickness figures elsewhere in this file), the
# "thin region" is most of the model, not a few spike tips. Growing that by
# enough depth to actually move the needle (radius=3 voxels, matching the
# floor) inflated Godzilla's volume by >1000% and visibly changed its
# silhouette (bounding box shrank on one axis — a sign shell selection
# changed too, not just surface puffing); capping the growth to 1 voxel
# tamed the volume blowup only to ~475% while barely moving the measured
# below-floor fraction at all (shallow growth on such a large contiguous
# area still adds a lot of total volume, just not enough depth to help).
# Left implemented (not deleted) as a documented dead end and a base for a
# smarter version later — e.g. restricting it to small, genuinely-isolated
# thin *connected components* (a claw, a single spike) via connected-
# component size filtering, rather than the whole thin-relative-to-1.2mm
# surface at once — but that's real additional work, not a config flag.
# Matches Meshy's own admitted difficulty here (research/meshy-ai-
# research.md §5.2: their CEO describes wall-thickness/hollowing as still
# unsolved roadmap items even for them) — this is a genuinely hard problem,
# not a quick fix.
MIN_FEATURE_THICKENING_ENABLED = False
# Skip thickening (rather than risk a multi-GB distance-transform) when the
# voxel grid at the current resolution has more cells than this. Well above
# what real tests here have hit (up to ~250M cells at the
# DETAIL_MAX_VOXEL_RESOLUTION=600 ceiling for a roughly cubic object), so
# this should rarely trigger — a resolution of 600 is only reached for
# large (~200mm+) prints in the first place.
MIN_FEATURE_THICKENING_MAX_VOXEL_CELLS = 300_000_000
# How far (in voxels) to actually grow a detected-thin region, decoupled
# from the radius used to *detect* thinness above. Empirically necessary:
# using the full detection radius as the dilation distance too (the
# naive/obvious choice) grew Godzilla's volume by >1000% and visibly
# changed its silhouette — on a real organic sculpt, the vast majority of
# the surface skin already reads as "thin" relative to a full 1.2mm floor
# (matches the ~80-99% below-floor figures noted elsewhere in this file),
# so dilating *all* of that by a full 3 voxels inflates the entire model
# like a balloon rather than just fattening isolated spike/claw tips. A
# small fixed cap makes this a modest, localized nudge instead of a full
# guarantee — see MIN_FEATURE_THICKENING_ENABLED's docstring-comment for
# the tradeoff this implies.
MIN_FEATURE_THICKENING_MAX_DILATION_VOXELS = 1

# --- Text-to-image stage (mflux / Z-Image Turbo, isolated `uv tool install`) ---
# Installed outside this project's venv (see pyproject.toml comment in
# pipeline/stages/text_to_image/mflux_stage.py) since mflux requires
# transformers>=5.0, which conflicts with SF3D's pinned transformers==4.42.3.
MFLUX_EXECUTABLE = "mflux-generate-z-image-turbo"
# 20 (not the turbo model's minimal 9) — visibly crisper feather/edge detail
# in a direct same-seed comparison, for ~10s extra cost. Note: bumping
# IMAGE_GEN_SIZE_PX above 512 would NOT help despite the temptation — verified
# in trellis2mlx/generate.py that _extract_image_features() always resizes
# to exactly 512px before DINOv3 conditioning, so a larger source image is
# downscaled right back down and wastes generation time for zero benefit.
IMAGE_GEN_STEPS = 20
IMAGE_GEN_SIZE_PX = 512
IMAGE_GEN_QUANTIZE_BITS = 8
# Appended to every user prompt: SF3D's reconstruction quality is sensitive to
# input-image style (see research/local-text-to-3d-models-m4-max.md, Phase 2
# risk) — a clean, isolated, well-lit single object reconstructs far better
# than a busy scene.
#
# The trailing "thick-walled, sturdy, solid form" is a printability steer
# (2026-09-18), sourced from Tripo3D's own printability blog post (see
# research/meshy-ai-research.md §7) as a disclosed lever their service uses.
# Tested directly here, same seed/scale, prompt as the only variable: on a
# real owl-figurine generation, the fraction of sampled surface below the
# 1.2mm FDM wall-thickness floor dropped from 83.0% to 68.8% (below-one-
# nozzle-width from 77.5% to 68.3%), with zero extra generation cost. Single
# data point (n=1) — not validated across other subjects/seeds yet — but the
# downside risk of a few extra prompt words is effectively zero, so it's on
# by default rather than gated behind a flag.
IMAGE_GEN_PROMPT_SUFFIX = (
    ", single object, centered, white background, product photo, studio lighting"
    ", thick-walled, sturdy, solid form"
)

# --- Image-to-3D: best-of-N candidate selection (2026-09-18) ---------------
# TRELLIS.2's reconstruction is image-conditioned, not free-form — different
# seeds on the identical input image were verified to produce visually
# near-identical shapes (same silhouette/pose/spike layout in a direct
# rendered comparison) but measurably different *reconstruction cleanliness*:
# two seeds on the same real test image gave boundary-edge counts of 13,623
# vs. 6,855 and wall-thickness below-floor fractions of 87.0% vs. 22.2% — a
# far bigger lever than any single generation parameter tuned this session
# (doubling --steps only moved non-manifold edges ~20% at ~2x the time cost).
# Since generation is fully local with no per-run cost, generating several
# candidates and keeping the cleanest (by topology-defect count — see
# trellis2_stage.py's _topology_defect_count) captures that benefit
# automatically. 3 was chosen as a default balance between the clear quality
# win and tripling generation time (~3.5-4min -> ~10-12min per job); still
# overridable per-request (CLI --candidates, web UI control).
TRELLIS2_NUM_CANDIDATES = 3

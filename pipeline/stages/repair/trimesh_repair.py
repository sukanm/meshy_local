"""Print-readiness stage: the first-class, deliberately engineered pipeline
step called for in CLAUDE.md and research/meshy-ai-research.md section 5.

Escalation ladder (each tier only runs if the previous one didn't achieve a
watertight result), discovered empirically against real SF3D output (a
generated owl figurine mesh needed all the way to tier 3):

  1. trimesh baseline repair (always run, silent): hole filling, normal/
     winding correction, duplicate/degenerate face removal. Safe,
     unambiguous operations with no risk of distorting intent.
  2. PyMeshLab topology repair (disclosed): dedicated non-manifold-edge/
     vertex repair and hole closing. Real repair filters, not just
     validation — this is what actually closes many holes trimesh's own
     heuristics can't.
  3. Voxel remeshing via marching cubes (disclosed, stronger warning):
     mathematically guaranteed watertight by construction, at the cost of
     losing fine surface detail. The reliable last resort when 1-2 aren't
     enough (empirically the case for un-retouched SF3D output). Followed
     by a detail-recovery pass: each vertex is snapped to the nearest point
     on the pre-escalation reference mesh, restoring fine surface texture
     (scute/scale patterns, individual claws/teeth) that marching cubes at
     any practical resolution plus smoothing otherwise erases. This only
     moves vertex *positions* — it cannot change the mesh's topology/
     connectivity, so it cannot un-watertight an already-watertight mesh.
     (Verified directly, not assumed: a wildly negative euler number was
     initially mistaken for tangling introduced by the snap, but a control
     test showed that exact number is already present before smoothing and
     before snapping — it's this shape's actual topology, an organic
     multi-spiked form with many small loops, unrelated to the snap. The
     real, airtight validity check is a direct edge-multiplicity count:
     zero boundary edges and zero non-manifold edges, both before and after
     snapping. Do NOT run merge_vertices()/nondegenerate_faces() cleanup
     after snapping — empirically this *damages* an already-valid result
     by merging vertices that are close in space but meaningfully distinct
     in the fine-detail regions the snap just recovered.)

     However: STL is a triangle-soup format (see
     pipeline/stages/export/mesh_topology.py), so any real consumer must
     reconstruct topology by welding vertices at (near-)identical
     *positions*, not by trusting our in-memory indices. On some real
     meshes — a densely feather-textured owl figure, not just the
     originally-diagnosed spiky dragon — the fine-detail positions the snap
     introduces are hard to correctly re-weld this way: position-based
     welding reconstructs a topology with hundreds to low-thousands of
     scattered non-manifold/boundary edges, and (discovered the hard way)
     iteratively removing/refilling faces around *many scattered* defects
     systematically makes the boundary-edge count *worse*, not better,
     regardless of iteration budget — each removal cascades into more holes
     than can be re-closed. So the snap's result is validated the same way
     STLExportStage's weld_and_repair will see it *before* being committed
     to: if it has more than a small number of residual defects, the snap
     is discarded and the smoothed-but-unsnapped mesh (unconditionally
     valid — marching cubes guarantees this, and it needs no
     position-based re-welding trickery) is used instead. Detail recovery
     is a bonus applied when safe, not something ever allowed to risk
     shipping a defective STL.

Note on manifold3d: evaluated and rejected for this escalation ladder.
Its `Manifold(mesh)` constructor *validates* that input is already an
oriented 2-manifold and returns an empty result with `Error.NotManifold`
otherwise — it does not repair non-manifold geometry, so it can't serve as
a repair step here (see research/local-text-to-3d-models-m4-max.md and the
build log for the empirical test that established this).

Multi-shell handling and wall-thickness/build-volume checks follow the
auto-fix/disclose/flag split from CLAUDE.md's plan: see inline comments at
each decision point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from pipeline import config
from pipeline.stages.repair.thickness import estimate_min_wall_thickness_mm
from pipeline.types import Issue, PrintReadyMesh, PrintReadinessStatus, RawMesh, RepairReport, Severity


def _baseline_repair(mesh: trimesh.Trimesh) -> None:
    """Safe, unambiguous fixes. Mutates in place."""
    mesh.process(validate=True)
    mesh.fill_holes()
    mesh.fix_normals()
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()


def _pymeshlab_repair(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Tier 2 escalation. Returns a new mesh; does not mutate the input."""
    import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=mesh.vertices,
            face_matrix=mesh.faces.astype(np.int32),
        )
    )
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_null_faces()
    # Non-manifold repair MUST run before re-orient: re_orient_faces_coherently
    # requires the mesh already be manifold and raises otherwise (this was
    # backwards before — the whole point of the repair calls is to fix
    # non-manifoldness ahead of re-orientation, not after. Caught on real
    # TRELLIS.2 output, which crashed here until this was reordered.)
    ms.meshing_repair_non_manifold_edges()
    ms.meshing_repair_non_manifold_vertices()
    ms.meshing_re_orient_faces_coherently()
    ms.meshing_close_holes(maxholesize=1000)

    repaired = ms.current_mesh()
    return trimesh.Trimesh(
        vertices=repaired.vertex_matrix(), faces=repaired.face_matrix(), process=True
    )


def _curvature_adaptive_smooth(mesh: trimesh.Trimesh, iterations: int) -> trimesh.Trimesh:
    """Blends Taubin-smoothed vertex positions back toward the raw (pre-
    smoothing) marching-cubes positions in high-curvature regions (teeth,
    claws, fingertips), instead of applying one uniform smoothing strength
    everywhere. Added 2026-09-18 after the user reported facial features and
    fingers were "too smooth" — plain uniform smoothing rounds off exactly
    the fine features the detail-recovery snap exists to restore; this
    reduces how much needs recovering in the first place, and leaves flat/
    low-curvature regions (torso, limbs) fully smoothed since they have no
    fine detail to lose and benefit most from staircase removal.

    Curvature is `mesh.vertex_defects` (discrete angle defect) on the
    pre-smoothing mesh — purely combinatorial from face angles around each
    vertex, needs no extra radius/scale parameter, and is near-zero on flat
    regions, large at sharp points/edges: exactly what's wanted here.

    Returns a fresh Trimesh (never mutates in place) — see the module's
    established caution around trimesh's cached properties not reliably
    invalidating on in-place vertex mutation.
    """
    if iterations <= 0:
        return mesh

    original_vertices = mesh.vertices.copy()
    curvature = np.abs(mesh.vertex_defects)

    smoothed = trimesh.Trimesh(vertices=original_vertices.copy(), faces=mesh.faces, process=False)
    trimesh.smoothing.filter_taubin(smoothed, lamb=0.5, nu=-0.53, iterations=iterations)

    if curvature.max() > 0:
        # Percentile, not the raw max: a single extreme-curvature vertex
        # (e.g. one claw tip) would otherwise saturate the whole
        # normalization and make every other genuinely-detailed vertex look
        # "flat" by comparison.
        cap = max(float(np.percentile(curvature, config.CURVATURE_SMOOTHING_PERCENTILE_CAP)), 1e-9)
        normalized = np.clip(curvature / cap, 0.0, 1.0)
    else:
        normalized = np.zeros_like(curvature)

    # How much of the pre-smoothing position survives at each vertex, scaled
    # so even the highest-curvature vertices still get some smoothing (fully
    # skipping it would leave raw staircase artifacts on the sharpest
    # features, the opposite of what's wanted).
    keep_original = (normalized * (1.0 - config.CURVATURE_SMOOTHING_MIN_FACTOR))[:, None]
    blended = keep_original * original_vertices + (1.0 - keep_original) * smoothed.vertices

    return trimesh.Trimesh(vertices=blended, faces=mesh.faces.copy(), process=False)


def _thicken_thin_voxels(
    matrix: np.ndarray, pitch_mm: float, min_thickness_mm: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """Locally dilates regions of a boolean voxel occupancy grid that don't
    have `min_thickness_mm` of local thickness anywhere nearby, instead of
    only ever reporting thin walls as a warning after export (see
    config.MIN_FEATURE_THICKENING_ENABLED).

    Local thickness proxy: `distance_transform_edt` gives, for every solid
    voxel, its distance to the nearest background voxel — the radius of the
    largest ball centered there that still fits in the solid. A region has
    no "thick core" nearby (i.e. is thin *throughout*, not just at its
    surface skin) exactly when the local maximum of that distance, in a
    window sized to the target thickness, never reaches the target radius —
    that's true of every voxel in a thin spike/claw, and false near the
    center of anything thick enough already. Only those genuinely-thin
    voxels get dilated, not the whole mesh.

    Returns (thickened_matrix, added_mask), or (matrix, None) if nothing
    needed thickening (avoids the caller having to special-case "no changes"
    against a zeros-shaped mask it doesn't otherwise need).
    """
    from scipy import ndimage

    target_radius_voxels = max(1, round((min_thickness_mm / 2.0) / pitch_mm))
    dist_in = ndimage.distance_transform_edt(matrix)
    window = 2 * target_radius_voxels + 1
    local_max = ndimage.maximum_filter(dist_in, size=window)
    thin_mask = matrix & (local_max < target_radius_voxels)
    if not thin_mask.any():
        return matrix, None

    dilated_thin = ndimage.binary_dilation(thin_mask, iterations=config.MIN_FEATURE_THICKENING_MAX_DILATION_VOXELS)
    thickened = matrix | dilated_thin
    added_mask = thickened & ~matrix
    if not added_mask.any():
        return matrix, None
    return thickened, added_mask


def _vertices_near_mask(vertices: np.ndarray, transform: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Which of `vertices` (world-space) fall in or adjacent to a True cell
    of a voxel-grid boolean `mask` (grid-space, related to world space by
    `transform`) — used to keep the detail-recovery snap away from voxels
    that minimum-feature thickening just added, since snapping them back
    onto the original (thin) reference surface would silently undo the
    thickness guarantee for exactly the features it was added to protect.
    Dilated by one voxel as a safety margin around the boundary.
    """
    from scipy import ndimage

    padded = ndimage.binary_dilation(mask, iterations=1)
    inv = np.linalg.inv(transform)
    idx = np.round(trimesh.transformations.transform_points(vertices, inv)).astype(int)
    shape = np.array(padded.shape)
    valid = np.all((idx >= 0) & (idx < shape), axis=1)
    result = np.zeros(len(vertices), dtype=bool)
    idx_clipped = np.clip(idx, 0, shape - 1)
    result[valid] = padded[idx_clipped[valid, 0], idx_clipped[valid, 1], idx_clipped[valid, 2]]
    return result


def _voxel_remesh(
    mesh: trimesh.Trimesh,
    reference: trimesh.Trimesh | None = None,
    resolution: int = config.VOXEL_REMESH_RESOLUTION,
    smoothing_iterations: int = config.VOXEL_REMESH_SMOOTHING_ITERATIONS,
    snap_to_reference: bool = True,
    pitch_mm: float | None = None,
) -> tuple[trimesh.Trimesh, bool, float]:
    """Returns (mesh, detail_recovered, thickened_fraction). detail_recovered
    is False when the snap was attempted but discarded for having too many
    residual defects once welded the way an STL consumer would (see class
    docstring above). thickened_fraction is the fraction of solid voxels
    added by minimum-feature thickening (0.0 if disabled, skipped for size,
    or nothing needed it).

    Tier 3 escalation. Guaranteed watertight by construction (a marching-
    cubes surface over a filled voxel grid is inherently a closed
    2-manifold) — the reliable last resort, at the cost of fine detail.

    Followed by light Taubin smoothing: marching cubes over a voxel grid
    produces visible staircase/blocky artifacts on the surface (an axis-
    aligned grid can't represent an angled surface exactly). Taubin
    smoothing (alternating shrink/inflate passes) removes that faceting
    without the shrinkage plain Laplacian smoothing would cause, and only
    moves vertex positions — it can't change topology, so the mesh stays
    exactly as watertight as the marching-cubes output already was.

    smoothing_iterations=0 skips smoothing entirely (leaves the raw,
    blockier marching-cubes surface); higher values smooth more aggressively
    at the cost of rounding off fine features. User-tunable — see
    pipeline/config.py for the default and CLAUDE.md for how this was tuned
    against real output.
    """
    extent = mesh.bounds[1] - mesh.bounds[0]
    pitch = extent.max() / resolution
    vox = mesh.voxelized(pitch=pitch).fill()

    added_mask = None
    thickened_fraction = 0.0
    if config.MIN_FEATURE_THICKENING_ENABLED and pitch_mm is not None:
        if vox.matrix.size <= config.MIN_FEATURE_THICKENING_MAX_VOXEL_CELLS:
            original_solid_count = int(vox.matrix.sum())
            thickened_matrix, added_mask = _thicken_thin_voxels(
                vox.matrix, pitch_mm=pitch_mm, min_thickness_mm=config.FDM_MIN_WALL_THICKNESS_MM
            )
            if added_mask is not None:
                thickened_fraction = float(added_mask.sum()) / max(original_solid_count, 1)
                vox = trimesh.voxel.VoxelGrid(thickened_matrix, transform=vox.transform)
        # else: grid too large to risk a multi-GB distance transform — skip
        # thickening, still produce the (untouched) voxel-remesh output.

    remeshed = vox.marching_cubes
    remeshed.apply_transform(vox.transform)

    if len(remeshed.faces) > config.VOXEL_REMESH_MAX_FACES_BEFORE_SIMPLIFY:
        # Bound cost before any per-vertex step: see config.py for why this
        # can happen (surface texture density, not resolution, drives
        # marching-cubes output size).
        #
        # PyMeshLab's quadric edge-collapse, not fast_simplification: the
        # latter was tried first and is fast (~1-2s), but was found to
        # introduce real, severe topology damage on this exact kind of
        # input — a single clean 569K-face shell came out as 502 separate
        # shells with 837 non-manifold edges, and the damage was still
        # present (150 non-manifold edges, 27 shells) even at the most
        # conservative reduction tested (barely 30%, 569K->400K), so it
        # wasn't a matter of being too aggressive. PyMeshLab's
        # implementation with preservetopology=True on the exact same
        # input produces a single clean shell with zero defects (Godzilla
        # test case) or a small, easily-cleaned-up residual (30 defects,
        # 21 shells on a densely-textured owl case) — slower (2-18s here)
        # but the only one of the two that's actually safe to rely on.
        import pymeshlab

        ms = pymeshlab.MeshSet()
        ms.add_mesh(
            pymeshlab.Mesh(vertex_matrix=remeshed.vertices, face_matrix=remeshed.faces.astype(np.int32))
        )
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=config.VOXEL_REMESH_SIMPLIFY_TARGET_FACES,
            preservetopology=True,
            preserveboundary=True,
        )
        decimated = ms.current_mesh()
        remeshed = trimesh.Trimesh(
            vertices=decimated.vertex_matrix(), faces=decimated.face_matrix(), process=False
        )
        # Even PyMeshLab's topology-preserving decimation can leave a small
        # residual (see above) — cheap baseline cleanup immediately, before
        # smoothing/snapping/floater-removal have to deal with it.
        remeshed.process(validate=True)
        remeshed.fill_holes()
        remeshed.fix_normals()
        remeshed.merge_vertices()
        remeshed.update_faces(remeshed.nondegenerate_faces())
        remeshed.remove_unreferenced_vertices()

    # Sliver-edge cleanup: unconditional (not just after decimation above) —
    # see mesh_topology.find_sliver_edges for the real incident this was
    # diagnosed from (a user-reported Godzilla STL with long thin spikes
    # shooting out from the body). Runs before smoothing/snapping so neither
    # of those steps has to contend with (or gets blamed for) a degenerate
    # triangle that was already there.
    from pipeline.stages.export.mesh_topology import remove_slivers

    remeshed = remove_slivers(remeshed, config.SLIVER_EDGE_LENGTH_MULTIPLE)

    if smoothing_iterations > 0:
        remeshed = _curvature_adaptive_smooth(remeshed, smoothing_iterations)

    detail_recovered = False
    if (
        snap_to_reference
        and reference is not None
        and len(reference.faces) >= config.DETAIL_SNAP_MIN_REFERENCE_FACES
    ):
        closest, distance, _tri_id = reference.nearest.on_surface(remeshed.vertices)
        # Safety clamp: leave any pathologically-far vertex at its smoothed
        # position rather than snapping it — guards against a vertex whose
        # nearest reference point is, for whatever reason, not actually
        # local (e.g. a gap in the reference mesh near a floater that was
        # already removed earlier in the pipeline).
        max_snap = 0.05 * float(np.linalg.norm(extent))
        allow_snap = distance < max_snap
        if added_mask is not None:
            # Don't let the detail-recovery snap pull newly-thickened
            # material back down onto the original (thin) reference surface
            # — that would silently undo the thickness guarantee for
            # exactly the features it was added to protect.
            near_thickened = _vertices_near_mask(remeshed.vertices, vox.transform, added_mask)
            allow_snap = allow_snap & ~near_thickened
        move = np.where(allow_snap[:, None], closest, remeshed.vertices)
        # Build a *fresh* Trimesh rather than assigning `.vertices` in place:
        # trimesh's cached properties (is_watertight, euler_number, ...) do
        # not reliably invalidate on in-place vertex mutation, and a stale
        # cache here would silently hide a real problem. A fresh object
        # forces every check downstream to be recomputed from scratch.
        # Deliberately NOT followed by merge_vertices()/nondegenerate_faces()
        # cleanup — verified empirically that doing so damages an already-
        # valid result (see module docstring).
        snapped = trimesh.Trimesh(vertices=move, faces=remeshed.faces.copy(), process=False)

        # Validate — and actively repair — the way an actual STL consumer
        # will see it (position-welded), not just via in-memory topology.
        # Earlier versions of this just counted residual defects and
        # rejected the snap above a threshold; upgraded to actually calling
        # weld_and_repair after finding that its junction-vertex handling
        # (see mesh_topology.py) reliably closes exactly the kind of
        # defects the snap introduces — Godzilla's snap, previously
        # rejected at 225 boundary + 106 non-manifold edges, now repairs
        # cleanly in 5 iterations. Only fall back to the unsnapped mesh if
        # weld_and_repair itself can't fully close it.
        from pipeline.stages.export.mesh_topology import weld_and_repair

        repaired_snap, is_clean, _stats = weld_and_repair(snapped)
        if is_clean:
            remeshed = repaired_snap
            detail_recovered = True
        # else: keep the pre-snap `remeshed` (smoothed, unconditionally
        # valid) — the snap's detail isn't worth the validity risk here.

        # Second sliver pass: the snap itself can independently introduce a
        # new sliver (moving one vertex onto a messy/self-intersecting
        # reference surface without regard to how far its neighbors moved),
        # separate from the pre-decimation slivers the first pass above
        # catches. Confirmed necessary empirically — the owl test mesh
        # (this project's hardest case) still had a 27.6x-median sliver
        # after only the pre-snap pass. remove_slivers keeps the mesh
        # watertight via its own fan-fill, same as the first pass.
        remeshed = remove_slivers(remeshed, config.SLIVER_EDGE_LENGTH_MULTIPLE)

    return remeshed, detail_recovered, thickened_fraction


def _remove_small_floaters(mesh: trimesh.Trimesh) -> tuple[trimesh.Trimesh, int, float]:
    """Drops only shells that are small relative to total volume (generation
    noise), keeping every substantial shell — including a deliberately
    hollow object's inner cavity wall, which is topologically a *separate*
    shell (shares no edges with the outer surface) despite being an
    intentional, load-bearing part of the geometry, not a floater. Picking
    only the single largest shell would wrongly collapse a hollow print into
    a solid one; this was caught by test_thin_wall_flags_warning_without_auto_fixing
    during Phase 3 development.

    Returns (mesh_with_floaters_removed, num_discarded_shells, discarded_volume_fraction).
    """
    shells = mesh.split(only_watertight=False)
    if len(shells) <= 1:
        return mesh, 0, 0.0

    def _shell_volume(s: trimesh.Trimesh) -> float:
        if s.is_watertight:
            return abs(s.volume)
        try:
            return s.convex_hull.volume
        except Exception:  # noqa: BLE001
            # A degenerate sliver shell (too few/nearly-coplanar points for
            # qhull to build a hull — seen in practice after simplifying an
            # oversized voxel-remesh output) is, almost by definition, tiny
            # generation noise. Falling back to the bounding box still ranks
            # it correctly against real shells without crashing the stage.
            return float(np.prod(s.bounding_box.extents))

    volumes = [_shell_volume(s) for s in shells]
    total_volume = sum(volumes)
    if total_volume <= 0:
        return mesh, 0, 0.0

    kept = [
        s for s, v in zip(shells, volumes) if v / total_volume >= config.FLOATER_VOLUME_FRACTION_THRESHOLD
    ]
    discarded_fraction = 1.0 - (sum(
        v for v in volumes if v / total_volume >= config.FLOATER_VOLUME_FRACTION_THRESHOLD
    ) / total_volume)
    num_discarded = len(shells) - len(kept)

    if num_discarded == 0:
        return mesh, 0, 0.0
    result = kept[0] if len(kept) == 1 else trimesh.util.concatenate(kept)
    return result, num_discarded, discarded_fraction


class TrimeshRepairStage:
    def process(
        self,
        mesh: RawMesh,
        target_longest_dimension_mm: float,
        work_dir: Path,
        smoothing_iterations: int | None = None,
        voxel_resolution: int | None = None,
    ) -> PrintReadyMesh:
        if voxel_resolution is None:
            # Geometry/scale-aware, not a fixed constant: see
            # adaptive_voxel_resolution's docstring for why a fixed
            # resolution silently gives worse real-world detail for larger
            # prints. Still overridable (the web UI's mesh-detail slider,
            # or --detail on the CLI) for a user who wants to force a
            # specific value.
            voxel_resolution = config.adaptive_voxel_resolution(target_longest_dimension_mm)
        if smoothing_iterations is None:
            # Resolution-aware too, computed *after* resolution is resolved
            # above: a finer grid needs less smoothing to look clean (see
            # adaptive_smoothing_iterations's docstring) — a user-supplied
            # --detail without a matching --smoothing still gets a sensible
            # paired default instead of always falling back to 10.
            smoothing_iterations = config.adaptive_smoothing_iterations(voxel_resolution)
        work_dir.mkdir(parents=True, exist_ok=True)
        loaded = trimesh.load(mesh.mesh_path, force="mesh")
        # Strip loader-attached per-face/vertex attributes (e.g. STL's
        # per-face 'stl' flag): later ops like fill_holes() change face
        # counts, leaving those stale arrays mismatched and crashing export.
        # We only need geometry through this pipeline, never carry them.
        m = trimesh.Trimesh(vertices=loaded.vertices, faces=loaded.faces, process=False)

        actions: list[str] = []
        issues: list[Issue] = []

        _baseline_repair(m)
        actions.append("Baseline repair: filled small holes, fixed normals/winding, removed duplicate/degenerate faces")
        # Kept at full original detail, untouched by any lossy escalation
        # tier, specifically to snap voxel-remeshed vertices back onto if
        # tier 3 is needed — see _voxel_remesh's docstring.
        reference_mesh = m.copy()

        if not m.is_watertight:
            try:
                m = _pymeshlab_repair(m)
                actions.append(
                    "Escalated to PyMeshLab non-manifold-edge/vertex repair and hole closing "
                    "(mesh was still non-watertight after baseline repair)"
                )
            except Exception as exc:  # noqa: BLE001
                # PyMeshLab's own filters can require the mesh already be
                # manifold (e.g. meshing_re_orient_faces_coherently) and
                # raise rather than degrade gracefully on messier input
                # (observed on real TRELLIS.2 output). Fall through to the
                # voxel-remesh tier instead of crashing the whole stage.
                actions.append(
                    f"PyMeshLab repair failed ({exc}); falling through to voxel remeshing"
                )

        if not m.is_watertight:
            # Real-world (post-scale) voxel pitch, so minimum-feature
            # thickening targets FDM_MIN_WALL_THICKNESS_MM in actual mm, not
            # native mesh units — matches how adaptive_voxel_resolution
            # derives resolution from a target real-world pitch in the first
            # place (target_longest_dimension_mm / resolution ~=
            # DETAIL_TARGET_VOXEL_PITCH_MM when resolution isn't clamped).
            pitch_mm = target_longest_dimension_mm / voxel_resolution
            m, detail_recovered, thickened_fraction = _voxel_remesh(
                m,
                reference=reference_mesh,
                resolution=voxel_resolution,
                smoothing_iterations=smoothing_iterations,
                pitch_mm=pitch_mm,
            )
            if thickened_fraction > 0:
                actions.append(
                    f"Locally thickened {thickened_fraction * 100:.2f}% (by solid-voxel count) of thin "
                    f"features (spikes/claws/tips) to help meet the {config.FDM_MIN_WALL_THICKNESS_MM}mm "
                    "FDM wall-thickness floor, before it could be voxelized away by smoothing/snapping"
                )
            if detail_recovered:
                actions.append(
                    "Escalated to voxel remeshing (marching cubes) to force a watertight result, "
                    "then snapped vertices back onto the original detailed surface to recover fine "
                    "texture (resolution: {}, smoothing: {} iterations) — some detail loss possible "
                    "in areas where the original surface couldn't be closely matched".format(
                        voxel_resolution, smoothing_iterations
                    )
                )
            else:
                actions.append(
                    "Escalated to voxel remeshing (marching cubes) to force a watertight result "
                    f"(resolution: {voxel_resolution}, smoothing: {smoothing_iterations} iterations). "
                    "The detail-recovery step was attempted but discarded: it would have introduced "
                    "more topology defects than it's worth once re-welded the way a slicer would, "
                    "so this mesh keeps the smoothed (less detailed but guaranteed-valid) surface."
                )
            issues.append(
                Issue(
                    code="voxel_remesh_used",
                    message="Fine surface detail may be reduced: voxel remeshing was needed to "
                    "achieve a watertight mesh after lighter repair passes failed.",
                    severity=Severity.WARNING,
                )
            )

        is_watertight = m.is_watertight

        # Multi-shell handling: auto-keep largest shell, always disclosed
        # (changes the object, even though small floaters are near-certainly
        # generation noise for this MVP's single-object scope).
        m, num_discarded, discarded_fraction = _remove_small_floaters(m)
        if num_discarded > 0:
            actions.append(
                f"Removed {num_discarded} disconnected fragment(s) "
                f"({discarded_fraction * 100:.2f}% of total volume), kept the largest shell"
            )
            is_watertight = m.is_watertight

        # Scale to the user-specified target longest dimension. Uniform
        # scaling is a unit change, not a geometry change — always safe to
        # auto-apply (per CLAUDE.md's auto-fix/disclose/flag split).
        extent = m.bounds[1] - m.bounds[0]
        native_longest = float(extent.max())
        if native_longest <= 0:
            issues.append(
                Issue(
                    code="degenerate_mesh",
                    message="Mesh has zero or negative extent; cannot scale or validate.",
                    severity=Severity.BLOCKING,
                )
            )
            scale_to_mm = 1.0
        else:
            scale_to_mm = target_longest_dimension_mm / native_longest
            m.apply_scale(scale_to_mm)
            actions.append(
                f"Scaled uniformly so the longest dimension is {target_longest_dimension_mm:.1f}mm"
            )

        # Wall-thickness heuristic: report-only, never auto-fixed (thickening
        # geometry is a design decision, not a repair — see module docstring
        # and CLAUDE.md).
        min_thickness_mm = None
        below_floor_fraction = None
        critical_fraction = None
        if is_watertight and native_longest > 0:
            min_thickness_mm, below_floor_fraction, critical_fraction = estimate_min_wall_thickness_mm(
                m,
                scale_to_mm=1.0,  # mesh is already scaled to mm above
                sample_count=config.WALL_THICKNESS_SAMPLE_COUNT,
                floor_mm=config.FDM_MIN_WALL_THICKNESS_MM,
                critical_mm=config.WALL_THICKNESS_CRITICAL_MM,
            )
            if (
                below_floor_fraction is not None
                and below_floor_fraction > config.WALL_THICKNESS_WARNING_FRACTION
            ):
                # Tiered, not a single number: a knife-edge point (likely an
                # actual gap/hole once sliced) and a wall that's uniformly
                # 0.9mm (printable as a fragile 1-2 perimeter wall, just not
                # the robust 3-perimeter floor) used to get an identical
                # message — genuinely different severities in practice. See
                # config.WALL_THICKNESS_CRITICAL_MM for where the line is.
                fragile_fraction = below_floor_fraction - (critical_fraction or 0.0)
                if critical_fraction and critical_fraction > config.WALL_THICKNESS_WARNING_FRACTION:
                    breakdown = (
                        f"{critical_fraction * 100:.1f}% is below even one nozzle width "
                        f"({config.WALL_THICKNESS_CRITICAL_MM}mm) — likely to come out as actual "
                        f"gaps/holes once sliced, not just a fragile wall. The remaining "
                        f"{fragile_fraction * 100:.1f}% should still print as a thin (1-2 perimeter) "
                        "but present wall."
                    )
                else:
                    breakdown = (
                        "None of this is below one nozzle width, so it should still print as a "
                        "thin (1-2 perimeter) wall rather than an outright gap — just not the fully "
                        "robust 3-perimeter thickness."
                    )
                issues.append(
                    Issue(
                        code="thin_walls",
                        message=(
                            f"{below_floor_fraction * 100:.1f}% of sampled surface points have "
                            f"estimated wall thickness below the {config.FDM_MIN_WALL_THICKNESS_MM}mm "
                            f"FDM floor. {breakdown} Not auto-fixed — thickening geometry changes the "
                            "design. Consider a thicker prompt/scale, or manual repair in a slicer/CAD tool."
                        ),
                        severity=Severity.WARNING,
                    )
                )

        # Build-volume check against the target printer (flag only — never
        # silently resize past what the user explicitly requested).
        bbox_mm = tuple(float(x) for x in (m.bounds[1] - m.bounds[0]))
        fits_build_volume = all(
            dim <= limit for dim, limit in zip(bbox_mm, config.P2S_BUILD_VOLUME_MM)
        )
        if not fits_build_volume:
            issues.append(
                Issue(
                    code="exceeds_build_volume",
                    message=(
                        f"Bounding box {tuple(round(d, 1) for d in bbox_mm)}mm exceeds the "
                        f"Bambu Lab P2S build volume {config.P2S_BUILD_VOLUME_MM}mm. "
                        "Reduce --scale or split the model."
                    ),
                    severity=Severity.BLOCKING,
                )
            )

        # Overall status.
        if not is_watertight or any(i.severity == Severity.BLOCKING for i in issues):
            status = PrintReadinessStatus.NEEDS_MANUAL_REPAIR
        elif issues:
            status = PrintReadinessStatus.PRINT_READY_WITH_WARNINGS
        else:
            status = PrintReadinessStatus.PRINT_READY

        report = RepairReport(
            is_watertight=is_watertight,
            was_repaired=len(actions) > 1,  # baseline repair always runs; >1 means escalation happened
            actions_taken=actions,
            remaining_issues=issues,
            min_wall_thickness_mm=min_thickness_mm,
            wall_thickness_below_floor_fraction=below_floor_fraction,
            wall_thickness_critical_fraction=critical_fraction,
            bbox_mm=bbox_mm,
            fits_build_volume=fits_build_volume,
            status=status,
        )

        out_path = work_dir / "print_ready_mesh.ply"
        m.export(str(out_path))

        return PrintReadyMesh(mesh_path=out_path, repair_report=report)

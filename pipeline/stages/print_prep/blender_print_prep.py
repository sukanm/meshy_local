"""Print-prep stage backed by headless Blender (see pipeline/blender/).

Runs between repair and export, on the repair stage's already-watertight,
already-scaled, Z-up mesh:

  1. Wall offset (OpenVDB SDF, in Blender's bundled Python): grows the whole
     surface outward by r, so every wall becomes at least 2r thick. Auto by
     default — only applied when the repair stage measured a meaningful
     fraction of the surface below one nozzle width (walls likely to print
     as gaps or delaminate, as the first physical Godzilla print did).
     Measured on that Godzilla at r=0.6mm: below-floor 92.5% -> 1.0%,
     below-nozzle 80.0% -> 0.2%, ~1s. This softens fine detail and grows
     volume substantially (709 -> 7,696 mm^3 on that thin, spiky model) —
     disclosed in actions_taken, and overridable (0 = off) per job.

     Why OpenVDB and not the earlier candidates (CLAUDE.md has the full
     history): voxel-morphology dilation inflated volume >1000% while
     barely helping; CGAL alpha wrap took ~60s and overshot the theoretical
     maximum volume for a true offset by ~2x (it also bridges gaps between
     nearby features). An SDF offset is the textbook operation and lands
     right at that bound. It needs closed input (inside/outside comes from
     flood fill), which is why it runs here, after repair, and never on raw
     generator output.

  2. Base (Blender's Manifold boolean solver): "flatten" slices a thin layer
     off the lowest point for a real first-layer footprint; "plinth" unions
     a solid disc under the model; "none" skips.

  3. Placement: centred in XY, resting on Z=0.

Every Blender step falls back to the unmodified mesh with a WARNING if
Blender is missing, errors, or produces a non-closed result — print prep
improves printability when it can; it never fails a job.
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from pipeline import blender, config
from pipeline.stages.export.mesh_topology import _boundary_and_nonmanifold_edges
from pipeline.stages.repair.thickness import estimate_min_wall_thickness_mm, thin_wall_issue
from pipeline.stages.repair.trimesh_repair import remove_small_floaters
from pipeline.types import Issue, PrintReadyMesh, RepairReport, Severity, derive_status

_BLENDER_ERRORS = (blender.BlenderUnavailableError, blender.BlenderScriptError)


def _is_closed_manifold(mesh: trimesh.Trimesh) -> bool:
    """Direct edge-multiplicity check (every edge shared by exactly two
    faces) plus positive volume — not trimesh's cached is_watertight, per the
    stale-cache incident in CLAUDE.md."""
    if len(mesh.faces) == 0:
        return False
    boundary, nonmanifold = _boundary_and_nonmanifold_edges(mesh)
    # bool(): mesh.volume > 0 is a numpy.bool_, which json can't serialize —
    # it crashed job.json saving in the first real web-UI run.
    return bool(len(boundary) == 0 and len(nonmanifold) == 0 and mesh.volume > 0)


def _count_enclosed_shells(mesh: trimesh.Trimesh) -> int:
    """Shells whose bounding box lies strictly inside another shell's — a
    cheap proxy for sealed internal cavities (a hollow object's inner wall)."""
    shells = mesh.split(only_watertight=False)
    if len(shells) < 2:
        return 0
    bounds = np.array([sh.bounds for sh in shells])  # (n, 2, 3)
    count = 0
    for i, (lo, hi) in enumerate(bounds):
        others = np.delete(bounds, i, axis=0)
        if ((others[:, 0] < lo).all(axis=1) & (others[:, 1] > hi).all(axis=1)).any():
            count += 1
    return count


def resolve_wall_offset(requested_mm: float | None, report: RepairReport) -> tuple[float, str]:
    """Returns (offset_mm, reason). `requested_mm` None means auto."""
    if requested_mm is not None:
        if requested_mm <= 0:
            return 0.0, "off"
        return float(requested_mm), "requested"
    critical = report.wall_thickness_critical_fraction
    if critical is not None and critical > config.WALL_OFFSET_AUTO_TRIGGER_CRITICAL_FRACTION:
        return config.WALL_OFFSET_AUTO_MM, "auto"
    return 0.0, "auto-not-needed"


class BlenderPrintPrepStage:
    def prepare(
        self,
        mesh: PrintReadyMesh,
        work_dir: Path,
        wall_offset_mm: float | None = None,
        base_mode: str | None = None,
    ) -> PrintReadyMesh:
        base_mode = base_mode or config.BASE_MODE_DEFAULT
        if base_mode not in config.BASE_MODES:
            raise ValueError(f"Unknown base_mode {base_mode!r}; expected one of {config.BASE_MODES}")
        work_dir.mkdir(parents=True, exist_ok=True)

        report = copy.deepcopy(mesh.repair_report)
        actions = report.actions_taken
        issues = report.remaining_issues
        # process=False: this mesh was validated by the repair stage; default
        # on-load processing has damaged validated meshes before (CLAUDE.md).
        m = trimesh.load(mesh.mesh_path, force="mesh", process=False)

        offset_mm, offset_reason = resolve_wall_offset(wall_offset_mm, report)
        offset_applied = False
        offset_failed = False
        geometry_changed = False

        with tempfile.TemporaryDirectory(prefix="print_prep_") as tmp_name:
            tmp = Path(tmp_name)

            if offset_mm > 0:
                if not report.is_watertight:
                    issues.append(
                        Issue(
                            code="wall_offset_skipped",
                            message="Wall thickening skipped: it needs a watertight mesh, and this one isn't.",
                            severity=Severity.WARNING,
                        )
                    )
                    offset_failed = True
                else:
                    try:
                        m, note = self._wall_offset(m, offset_mm, tmp)
                        offset_applied = True
                        geometry_changed = True
                        how = "auto: walls were too thin to print reliably" if offset_reason == "auto" else "requested"
                        actions.append(
                            f"Thickened walls by a uniform {offset_mm:.2f}mm outward surface offset ({how}) — "
                            f"every wall is now at least {2 * offset_mm:.1f}mm thick; fine surface detail is "
                            f"softened and the model grows ~{2 * offset_mm:.1f}mm in each dimension.{note}"
                        )
                    except (*_BLENDER_ERRORS, _PrepResultInvalid) as exc:
                        offset_failed = True
                        issues.append(
                            Issue(
                                code="wall_offset_skipped",
                                message=f"Wall thickening couldn't run, so walls are unchanged: {exc}",
                                severity=Severity.WARNING,
                            )
                        )
            elif offset_reason == "auto-not-needed" and report.wall_thickness_critical_fraction is not None:
                actions.append(
                    "Wall thickening (auto) not applied: "
                    f"only {report.wall_thickness_critical_fraction * 100:.1f}% of the surface is below one "
                    "nozzle width"
                )

            if base_mode != "none":
                if not report.is_watertight:
                    issues.append(
                        Issue(
                            code="base_skipped",
                            message=f"Base '{base_mode}' skipped: the boolean needs a watertight mesh.",
                            severity=Severity.WARNING,
                        )
                    )
                else:
                    try:
                        m = self._base(m, base_mode, tmp)
                        geometry_changed = True
                        if base_mode == "flatten":
                            actions.append(
                                f"Flattened the bottom ({config.FLATTEN_BASE_DEPTH_MM}mm sliced off the lowest "
                                "point) for a flat first-layer footprint"
                            )
                        else:
                            actions.append(
                                f"Added a {config.PLINTH_THICKNESS_MM}mm plinth under the model's footprint "
                                f"(model sunk {config.PLINTH_EMBED_MM}mm into it)"
                            )
                    except (*_BLENDER_ERRORS, _PrepResultInvalid) as exc:
                        issues.append(
                            Issue(
                                code="base_skipped",
                                message=f"Base '{base_mode}' couldn't be added: {exc}",
                                severity=Severity.WARNING,
                            )
                        )

        # Placement: centred in XY, resting on the plate. Pure translation.
        lo, hi = m.bounds
        m.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])

        if geometry_changed:
            report.is_watertight = _is_closed_manifold(m)
            if report.is_watertight:
                (
                    report.min_wall_thickness_mm,
                    report.wall_thickness_below_floor_fraction,
                    report.wall_thickness_critical_fraction,
                ) = estimate_min_wall_thickness_mm(
                    m,
                    scale_to_mm=1.0,
                    sample_count=config.WALL_THICKNESS_SAMPLE_COUNT,
                    floor_mm=config.FDM_MIN_WALL_THICKNESS_MM,
                    critical_mm=config.WALL_THICKNESS_CRITICAL_MM,
                )

        # Re-issue the thin-wall warning with advice matching what happened.
        if offset_applied:
            advice = (
                f"Still present after a {offset_mm:.2f}mm wall offset — a larger custom offset or print "
                "scale would thicken further."
            )
        elif offset_failed:
            advice = "Wall thickening couldn't run (see the wall_offset_skipped warning)."
        elif offset_reason == "off":
            advice = "Wall thickening was turned off for this job; turn it on (auto) or print larger."
        else:
            advice = (
                "Auto wall thickening wasn't triggered because little of this is below one nozzle "
                f"width; set a custom offset (e.g. {config.WALL_OFFSET_AUTO_MM:.1f}mm) to reach the full floor."
            )
        issues[:] = [i for i in issues if i.code not in ("thin_walls", "exceeds_build_volume")]
        thin = thin_wall_issue(
            report.wall_thickness_below_floor_fraction, report.wall_thickness_critical_fraction, advice
        )
        if thin is not None:
            issues.append(thin)

        report.bbox_mm = tuple(float(x) for x in (m.bounds[1] - m.bounds[0]))
        report.fits_build_volume = bool(all(d <= lim for d, lim in zip(report.bbox_mm, config.P2S_BUILD_VOLUME_MM)))
        if not report.fits_build_volume:
            issues.append(
                Issue(
                    code="exceeds_build_volume",
                    message=(
                        f"Bounding box {tuple(round(d, 1) for d in report.bbox_mm)}mm exceeds the Bambu Lab "
                        f"P2S build volume {config.P2S_BUILD_VOLUME_MM}mm (after print prep). Reduce the "
                        "scale, or use a smaller base/offset."
                    ),
                    severity=Severity.BLOCKING,
                )
            )
        report.status = derive_status(report.is_watertight, issues)

        out_path = work_dir / "prepared_mesh.ply"
        m.export(str(out_path))
        return PrintReadyMesh(mesh_path=out_path, repair_report=report)

    @staticmethod
    def _wall_offset(m: trimesh.Trimesh, offset_mm: float, tmp: Path) -> tuple[trimesh.Trimesh, str]:
        voxel = config.wall_offset_voxel_mm(float(m.extents.max()))
        blender.run_script(
            "sdf_offset.py",
            blender.save_npz(m, tmp / "offset_in.npz"),
            tmp / "offset_out.npz",
            voxel,
            offset_mm,
            config.WALL_OFFSET_ADAPTIVITY,
        )
        result = blender.load_npz(tmp / "offset_out.npz")
        # The offset seals small concave pockets into tiny internal voids
        # (separate, inward-facing shells) — artifacts, not design intent.
        # Same volume-fraction rule the repair stage uses for floaters.
        result, n_dropped, _fraction = remove_small_floaters(result)
        if not _is_closed_manifold(result):
            raise _PrepResultInvalid("the offset result wasn't a closed manifold")
        notes = []
        n_cavities = _count_enclosed_shells(m)
        if n_cavities:
            # OpenVDB decides inside/outside by flood fill from the exterior,
            # so a sealed internal cavity is "inside" and comes out solid.
            # Found by the hollow-box test fixture. Harmless to print, but it
            # adds material and changes the design, so it's disclosed.
            notes.append(
                f" Filled {n_cavities} sealed internal cavit{'y' if n_cavities == 1 else 'ies'} solid "
                "(the offset treats enclosed voids as interior), adding material."
            )
        if n_dropped:
            notes.append(f" Removed {n_dropped} tiny sealed internal pocket(s) the offset created.")
        return result, "".join(notes)

    @staticmethod
    def _base(m: trimesh.Trimesh, base_mode: str, tmp: Path) -> trimesh.Trimesh:
        if base_mode == "flatten":
            params = [config.FLATTEN_BASE_DEPTH_MM]
        else:
            params = [config.PLINTH_THICKNESS_MM, config.PLINTH_EMBED_MM, config.PLINTH_MARGIN_MM]
        blender.run_script(
            "base_boolean.py",
            blender.save_npz(m, tmp / "base_in.npz"),
            tmp / "base_out.npz",
            base_mode,
            *params,
        )
        result = blender.load_npz(tmp / "base_out.npz")
        if not _is_closed_manifold(result):
            raise _PrepResultInvalid("the boolean result wasn't a closed manifold")
        if base_mode == "flatten" and not np.isclose(result.bounds[0][2], m.bounds[0][2] + config.FLATTEN_BASE_DEPTH_MM, atol=1e-3):
            raise _PrepResultInvalid("the flatten cut didn't land at the expected height")
        return result


class _PrepResultInvalid(RuntimeError):
    """A Blender step ran but its output failed validation."""

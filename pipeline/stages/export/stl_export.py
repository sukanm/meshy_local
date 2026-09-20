"""STL export stage: writes the repaired, already-scaled mesh to STL and
re-measures the *exported file's own* bounding box as a sanity check against
unit-conversion bugs (e.g. a format silently interpreted in meters instead
of millimeters) — see pipeline/stages/export/base.py docstring.
"""

from __future__ import annotations

from pathlib import Path

import trimesh

from pipeline.stages.export.mesh_topology import weld_and_repair
from pipeline.types import Issue, PrintReadyMesh, STLResult, Severity

# Loose tolerance: this is a sanity check against gross unit bugs (e.g. a
# 1000x meters/mm mismatch), not a precision assertion — small differences
# from STL's own lack of a units field and floating-point export are normal.
_BBOX_SANITY_TOLERANCE_FRACTION = 0.02


class STLExportStage:
    def export(self, mesh: PrintReadyMesh, work_dir: Path) -> STLResult:
        work_dir.mkdir(parents=True, exist_ok=True)
        # process=False is load-bearing here, not a style choice: trimesh's
        # default on-load processing (vertex merging/dedup) can silently
        # re-introduce non-manifold edges into an already-valid mesh — a
        # real bug caught during development where a mesh that went through
        # the repair stage's detail-recovery snap (see trimesh_repair.py)
        # came out with 1.6M non-manifold edges purely from being re-loaded
        # here with default settings and re-exported. The repair stage
        # already validated this mesh; this stage must not re-process it.
        m = trimesh.load(mesh.mesh_path, force="mesh", process=False)

        out_path = work_dir / "model.stl"
        m.export(str(out_path), file_type="stl")

        report = mesh.repair_report

        # STL is a triangle-soup format with no shared-index topology at
        # all — a real slicer, like any STL consumer, must reconstruct
        # the mesh's topology by welding vertices at (near-)identical
        # positions. That welding step can itself introduce a small number
        # of non-manifold/boundary edges even in a mesh that was verified
        # perfectly watertight before export (observed on real TRELLIS.2
        # output that went through the repair stage's detail-recovery
        # vertex-snap). Verify against — and if needed, repair and
        # re-export — the mesh as it will actually be consumed, not just
        # as it exists in memory right after our own export call.
        #
        # Single pass, deliberately: an earlier attempt at looping this
        # (re-check the re-exported file, retry weld_and_repair again if
        # the float32 STL round-trip introduced new defects) had a real bug
        # — the outer re-check loaded the STL without welding it first, so
        # it saw a bogus "every edge is a boundary edge" triangle-soup
        # reading and re-ran the repair+re-export cycle needlessly, which
        # measurably made real meshes *worse* by compounding float32
        # quantization noise across repeated exports. weld_and_repair
        # already does its own welding correctly internally; one pass is
        # what's actually validated to help, not hurt.
        welded = trimesh.load(out_path, force="mesh", process=False)
        repaired, is_clean, weld_stats = weld_and_repair(welded)
        if weld_stats["initial_boundary"] or weld_stats["initial_nonmanifold"]:
            repaired.export(str(out_path), file_type="stl")
            severity = Severity.WARNING if is_clean else Severity.BLOCKING
            outcome = "closed" if is_clean else "only partially closed"
            report.remaining_issues.append(
                Issue(
                    code="stl_topology_repaired" if is_clean else "stl_topology_unresolved",
                    message=(
                        f"The exported STL had {weld_stats['initial_boundary']} boundary and "
                        f"{weld_stats['initial_nonmanifold']} non-manifold edges once vertices were "
                        f"welded by position (as a slicer would) — {outcome} automatically after "
                        f"{weld_stats['iterations']} repair pass(es)."
                    ),
                    severity=severity,
                )
            )

        reloaded = trimesh.load(out_path, force="mesh", process=False)
        exported_bbox_mm = tuple(float(x) for x in (reloaded.bounds[1] - reloaded.bounds[0]))

        expected_bbox_mm = report.bbox_mm
        if expected_bbox_mm is not None:
            mismatch = any(
                abs(exp - got) > exp * _BBOX_SANITY_TOLERANCE_FRACTION + 1e-6
                for exp, got in zip(expected_bbox_mm, exported_bbox_mm)
            )
            if mismatch:
                report.remaining_issues.append(
                    Issue(
                        code="export_bbox_mismatch",
                        message=(
                            f"Exported STL bounding box {tuple(round(v, 2) for v in exported_bbox_mm)}mm "
                            f"doesn't match the repair stage's measurement "
                            f"{tuple(round(v, 2) for v in expected_bbox_mm)}mm by more than "
                            f"{_BBOX_SANITY_TOLERANCE_FRACTION * 100:.0f}% — possible unit-conversion bug."
                        ),
                        severity=Severity.BLOCKING,
                    )
                )

        return STLResult(stl_path=out_path, repair_report=report)

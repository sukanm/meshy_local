"""Produces a real, fully-validated STL using CGAL alpha wrap (via PyMeshLab)
as a true surface offset applied to the already-detailed, already-clean
print_ready_mesh.ply of a source job -- see scripts/spike_alpha_wrap_offset_detailed.py
for the experiment this is built from.

Writes a real output/<job_id>/ directory using the same STLExportStage path
production jobs use, so the resulting STL has gone through the same final
re-validation (weld-as-a-slicer-would-see-it) as every other job in output/.

Usage: uv run python -m scripts.export_alpha_wrap_stl <source_job_id> \
    [--alpha MM] [--offset MM]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import uuid

import pymeshlab
import trimesh

from pipeline import config
from pipeline.orchestrator import _json_default
from pipeline.stages.export.mesh_topology import weld_and_repair
from pipeline.stages.export.stl_export import STLExportStage
from pipeline.stages.repair.thickness import estimate_min_wall_thickness_mm
from pipeline.types import (
    Issue,
    Job,
    JobStatus,
    PrintReadinessStatus,
    PrintReadyMesh,
    RepairReport,
    Severity,
    TextPrompt,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_job_id", type=str)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--offset", type=float, default=0.6)
    args = parser.parse_args()

    source_dir = config.OUTPUT_DIR / args.source_job_id
    source_job = json.loads((source_dir / "job.json").read_text())
    source_ply = source_dir / "print_ready_mesh.ply"

    print(f"source={args.source_job_id} alpha={args.alpha}mm offset={args.offset}mm")

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(str(source_ply))
    ms.generate_alpha_wrap(
        alpha=pymeshlab.PureValue(args.alpha),
        offset=pymeshlab.PureValue(args.offset),
    )
    wrapped = ms.current_mesh()
    m = trimesh.Trimesh(vertices=wrapped.vertex_matrix(), faces=wrapped.face_matrix(), process=False)

    is_watertight = m.is_watertight
    repaired, is_clean, weld_stats = weld_and_repair(m)
    if is_clean:
        m = repaired
        is_watertight = True

    bbox_mm = tuple(float(x) for x in (m.bounds[1] - m.bounds[0]))
    min_thickness_mm = below_floor_fraction = critical_fraction = None
    issues: list[Issue] = []
    if is_watertight:
        min_thickness_mm, below_floor_fraction, critical_fraction = estimate_min_wall_thickness_mm(
            m, scale_to_mm=1.0, sample_count=config.WALL_THICKNESS_SAMPLE_COUNT,
            floor_mm=config.FDM_MIN_WALL_THICKNESS_MM, critical_mm=config.WALL_THICKNESS_CRITICAL_MM,
        )
        if below_floor_fraction and below_floor_fraction > config.WALL_THICKNESS_WARNING_FRACTION:
            issues.append(Issue(
                code="thin_walls",
                message=f"{below_floor_fraction*100:.1f}% below 1.2mm floor after alpha-wrap offset "
                        f"(alpha={args.alpha}mm, offset={args.offset}mm); {(critical_fraction or 0)*100:.1f}% below nozzle width.",
                severity=Severity.WARNING,
            ))
    issues.append(Issue(
        code="alpha_wrap_offset_used",
        message=(
            f"Experimental: CGAL alpha wrap (alpha={args.alpha}mm, offset={args.offset}mm) applied as a "
            "true surface offset to the already-detailed mesh, instead of the pipeline's normal "
            "detail-recovery snap. Preserves fine surface detail far better than voxel-grid dilation "
            "(this is a genuine uniform outward offset, not a re-voxelize+smooth round-trip), at the "
            "cost of substantial added volume/bulk (the whole model reads as uniformly thin, so a "
            "uniform offset over a large, spiky surface area adds a lot of material everywhere)."
        ),
        severity=Severity.INFO,
    ))

    fits_build_volume = all(dim <= limit for dim, limit in zip(bbox_mm, config.P2S_BUILD_VOLUME_MM))
    status = PrintReadinessStatus.PRINT_READY_WITH_WARNINGS if is_watertight else PrintReadinessStatus.NEEDS_MANUAL_REPAIR
    report = RepairReport(
        is_watertight=is_watertight,
        was_repaired=True,
        actions_taken=[f"CGAL alpha-wrap surface offset (alpha={args.alpha}mm, offset={args.offset}mm), applied to the detailed mesh"],
        remaining_issues=issues,
        min_wall_thickness_mm=min_thickness_mm,
        wall_thickness_below_floor_fraction=below_floor_fraction,
        wall_thickness_critical_fraction=critical_fraction,
        bbox_mm=bbox_mm,
        fits_build_volume=fits_build_volume,
        status=status,
    )

    job_id = uuid.uuid4().hex[:12]
    job_dir = config.OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True)
    ply_path = job_dir / "print_ready_mesh.ply"
    m.export(str(ply_path))

    job = Job(
        job_id=job_id,
        prompt=TextPrompt(**source_job["prompt"]),
        target_longest_dimension_mm=source_job["target_longest_dimension_mm"],
        status=JobStatus.RUNNING,
        stage_config={
            "rerun_from_source_job": args.source_job_id,
            "alpha_wrap_alpha_mm": args.alpha,
            "alpha_wrap_offset_mm": args.offset,
        },
        raw_mesh_path=source_job["raw_mesh_path"],
        repair_report=report,
    )

    def save() -> None:
        (job_dir / "job.json").write_text(json.dumps(dataclasses.asdict(job), default=_json_default, indent=2))

    save()

    stl_result = STLExportStage().export(PrintReadyMesh(mesh_path=ply_path, repair_report=report), job_dir)
    job.stl_path = stl_result.stl_path
    job.repair_report = stl_result.repair_report
    job.status = JobStatus.SUCCEEDED

    final_raw = trimesh.load(stl_result.stl_path, force="mesh", process=False)
    final_mesh, final_clean, _ = weld_and_repair(final_raw)
    job.repair_report.is_watertight = final_clean
    if final_clean:
        fmin, ffloor, fcrit = estimate_min_wall_thickness_mm(
            final_mesh, scale_to_mm=1.0, sample_count=config.WALL_THICKNESS_SAMPLE_COUNT,
            floor_mm=config.FDM_MIN_WALL_THICKNESS_MM, critical_mm=config.WALL_THICKNESS_CRITICAL_MM,
        )
        job.repair_report.min_wall_thickness_mm = fmin
        job.repair_report.wall_thickness_below_floor_fraction = ffloor
        job.repair_report.wall_thickness_critical_fraction = fcrit
    save()

    print(f"\nWrote {job.stl_path}")
    r = job.repair_report
    print(f"  watertight={r.is_watertight}  bbox_mm={tuple(round(x,2) for x in r.bbox_mm)}")
    print(f"  below 1.2mm floor: {(r.wall_thickness_below_floor_fraction or 0):.1%}   "
          f"below nozzle width: {(r.wall_thickness_critical_fraction or 0):.1%}")
    for issue in r.remaining_issues:
        print(f"  [{issue.severity.value}] {issue.code}: {issue.message}")


if __name__ == "__main__":
    main()

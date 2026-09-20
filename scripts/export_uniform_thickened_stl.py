"""Produces a real, fully-validated STL using uniform whole-mesh dilation
(see scripts/spike_uniform_thickening.py for the dose-response experiment
this is built from) instead of the pipeline's normal detail-recovery snap.

Writes a real output/<job_id>/ directory (job.json, print_ready_mesh.ply,
model.stl) using the same RepairReport/STLExportStage path production jobs
use, so the resulting STL has gone through the same final re-validation
(weld-as-a-slicer-would-see-it) as every other job in output/.

Usage: uv run python -m scripts.export_uniform_thickened_stl <source_job_id> \
    [--scale MM] [--depth N] [--resolution N]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import uuid

import numpy as np
import trimesh
from scipy import ndimage

from pipeline import config
from pipeline.orchestrator import _json_default
from pipeline.stages.export.mesh_topology import remove_slivers
from pipeline.stages.export.stl_export import STLExportStage
from pipeline.stages.repair.thickness import estimate_min_wall_thickness_mm
from pipeline.stages.repair.trimesh_repair import _curvature_adaptive_smooth, _remove_small_floaters
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
    parser.add_argument("--scale", type=float, default=None, dest="target_mm")
    parser.add_argument("--depth", type=int, default=3, help="Uniform dilation depth in voxels (default: 3, the sweet spot found in the dose-response spike).")
    parser.add_argument("--resolution", type=int, default=None)
    args = parser.parse_args()

    source_dir = config.OUTPUT_DIR / args.source_job_id
    source_job = json.loads((source_dir / "job.json").read_text())
    raw_mesh_path = source_job["raw_mesh_path"]
    target_mm = args.target_mm or source_job["target_longest_dimension_mm"]

    loaded = trimesh.load(raw_mesh_path, force="mesh")
    m = trimesh.Trimesh(vertices=loaded.vertices, faces=loaded.faces, process=False)
    m.merge_vertices()
    m.update_faces(m.unique_faces())
    m.fix_normals()

    native_extent = (m.bounds[1] - m.bounds[0]).max()
    scale_to_mm = target_mm / native_extent
    resolution = args.resolution or config.adaptive_voxel_resolution(target_mm)
    smoothing_iterations = config.adaptive_smoothing_iterations(resolution)
    extent = m.bounds[1] - m.bounds[0]
    pitch = extent.max() / resolution
    pitch_mm = pitch * scale_to_mm

    print(f"source={args.source_job_id} target_mm={target_mm} resolution={resolution} depth={args.depth} "
          f"(~{args.depth * pitch_mm:.3f}mm added radius)")

    vox = m.voxelized(pitch=pitch).fill()
    dilated = ndimage.binary_dilation(vox.matrix, iterations=args.depth) if args.depth > 0 else vox.matrix
    vox_d = trimesh.voxel.VoxelGrid(dilated, transform=vox.transform)

    remeshed = vox_d.marching_cubes
    remeshed.apply_transform(vox_d.transform)

    if len(remeshed.faces) > config.VOXEL_REMESH_MAX_FACES_BEFORE_SIMPLIFY:
        import pymeshlab

        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=remeshed.vertices, face_matrix=remeshed.faces.astype(np.int32)))
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=config.VOXEL_REMESH_SIMPLIFY_TARGET_FACES,
            preservetopology=True,
            preserveboundary=True,
        )
        d = ms.current_mesh()
        remeshed = trimesh.Trimesh(vertices=d.vertex_matrix(), faces=d.face_matrix(), process=False)
        remeshed.process(validate=True)
        remeshed.fill_holes()
        remeshed.fix_normals()
        remeshed.merge_vertices()
        remeshed.update_faces(remeshed.nondegenerate_faces())
        remeshed.remove_unreferenced_vertices()

    remeshed = remove_slivers(remeshed, config.SLIVER_EDGE_LENGTH_MULTIPLE)
    if smoothing_iterations > 0:
        remeshed = _curvature_adaptive_smooth(remeshed, smoothing_iterations)

    is_watertight = remeshed.is_watertight
    remeshed, num_discarded, discarded_fraction = _remove_small_floaters(remeshed)
    if num_discarded > 0:
        is_watertight = remeshed.is_watertight

    remeshed.apply_scale(scale_to_mm)

    min_thickness_mm = below_floor_fraction = critical_fraction = None
    issues: list[Issue] = []
    if is_watertight:
        min_thickness_mm, below_floor_fraction, critical_fraction = estimate_min_wall_thickness_mm(
            remeshed,
            scale_to_mm=1.0,
            sample_count=config.WALL_THICKNESS_SAMPLE_COUNT,
            floor_mm=config.FDM_MIN_WALL_THICKNESS_MM,
            critical_mm=config.WALL_THICKNESS_CRITICAL_MM,
        )
        if below_floor_fraction and below_floor_fraction > config.WALL_THICKNESS_WARNING_FRACTION:
            issues.append(
                Issue(
                    code="thin_walls",
                    message=(
                        f"{below_floor_fraction * 100:.1f}% of sampled surface points still below the "
                        f"{config.FDM_MIN_WALL_THICKNESS_MM}mm floor after uniform thickening "
                        f"(depth={args.depth}); {(critical_fraction or 0) * 100:.1f}% below nozzle width."
                    ),
                    severity=Severity.WARNING,
                )
            )
    issues.append(
        Issue(
            code="uniform_thickening_used",
            message=(
                f"Experimental: whole-mesh uniformly dilated by {args.depth} voxels "
                f"(~{args.depth * pitch_mm:.3f}mm added radius) to reach printable wall thickness. "
                "Fine surface detail (the usual detail-recovery snap) was NOT applied, since the "
                "existing snap-guard would block it everywhere on a fully-thickened mesh — this mesh "
                "trades detail for printability, it is not the pipeline's normal output."
            ),
            severity=Severity.INFO,
        )
    )

    bbox_mm = tuple(float(x) for x in (remeshed.bounds[1] - remeshed.bounds[0]))
    fits_build_volume = all(dim <= limit for dim, limit in zip(bbox_mm, config.P2S_BUILD_VOLUME_MM))
    status = (
        PrintReadinessStatus.PRINT_READY_WITH_WARNINGS
        if is_watertight
        else PrintReadinessStatus.NEEDS_MANUAL_REPAIR
    )
    report = RepairReport(
        is_watertight=is_watertight,
        was_repaired=True,
        actions_taken=[f"Uniform whole-mesh dilation, depth={args.depth} voxels, no detail-recovery snap"],
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
    remeshed.export(str(ply_path))

    job = Job(
        job_id=job_id,
        prompt=TextPrompt(**source_job["prompt"]),
        target_longest_dimension_mm=target_mm,
        status=JobStatus.RUNNING,
        stage_config={
            "rerun_from_source_job": args.source_job_id,
            "uniform_thickening_depth_voxels": args.depth,
            "voxel_resolution": resolution,
            "smoothing_iterations": smoothing_iterations,
        },
        raw_mesh_path=raw_mesh_path,
        repair_report=report,
    )

    def save() -> None:
        (job_dir / "job.json").write_text(json.dumps(dataclasses.asdict(job), default=_json_default, indent=2))

    save()

    stl_result = STLExportStage().export(PrintReadyMesh(mesh_path=ply_path, repair_report=report), job_dir)
    job.stl_path = stl_result.stl_path
    job.repair_report = stl_result.repair_report
    job.status = JobStatus.SUCCEEDED

    # The is_watertight/thickness numbers above were measured BEFORE the STL
    # export stage's own weld-as-a-slicer-would-see-it repair pass (which did
    # close a small residual: see the stl_topology_repaired issue above).
    # Re-measure on the actual final exported file, not the pre-repair
    # intermediate, so the reported numbers describe what a slicer will
    # actually load.
    from pipeline.stages.export.mesh_topology import weld_and_repair as _weld

    final_raw = trimesh.load(stl_result.stl_path, force="mesh", process=False)
    final_mesh, final_clean, _ = _weld(final_raw)
    job.repair_report.is_watertight = final_clean
    if final_clean:
        final_min, final_below_floor, final_critical = estimate_min_wall_thickness_mm(
            final_mesh,
            scale_to_mm=1.0,
            sample_count=config.WALL_THICKNESS_SAMPLE_COUNT,
            floor_mm=config.FDM_MIN_WALL_THICKNESS_MM,
            critical_mm=config.WALL_THICKNESS_CRITICAL_MM,
        )
        job.repair_report.min_wall_thickness_mm = final_min
        job.repair_report.wall_thickness_below_floor_fraction = final_below_floor
        job.repair_report.wall_thickness_critical_fraction = final_critical
    save()

    print(f"\nWrote {job.stl_path}")
    r = job.repair_report
    print(f"  watertight={r.is_watertight}  bbox_mm={tuple(round(x, 2) for x in r.bbox_mm)}")
    print(f"  below 1.2mm floor: {(r.wall_thickness_below_floor_fraction or 0):.1%}   "
          f"below nozzle width: {(r.wall_thickness_critical_fraction or 0):.1%}")
    for issue in r.remaining_issues:
        print(f"  [{issue.severity.value}] {issue.code}: {issue.message}")


if __name__ == "__main__":
    main()

"""Re-run repair+export on an already-generated raw mesh at a different
target scale, skipping the (multi-minute) text-to-image/image-to-3D stages.

Standing debug tool for the "does printing bigger fix thin-wall failures"
question: wall thickness in mm scales linearly with target_longest_dimension_mm,
so this is the cheapest way to A/B a job's printability across scales without
regenerating the mesh (and burning a different seed) each time.

Usage:
  uv run python -m scripts.rerun_repair_at_scale <source_job_id> --scale <mm> \
      [--smoothing N] [--detail N]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import uuid
from pathlib import Path

from pipeline import config
from pipeline.orchestrator import _json_default
from pipeline.stages.export.stl_export import STLExportStage
from pipeline.stages.repair.trimesh_repair import TrimeshRepairStage
from pipeline.types import Job, JobStatus, RawMesh, TextPrompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_job_id", type=str, help="Existing output/<job_id> to reuse raw_mesh.glb from.")
    parser.add_argument("--scale", type=float, required=True, dest="target_longest_dimension_mm")
    parser.add_argument("--smoothing", type=int, default=None, dest="smoothing_iterations")
    parser.add_argument("--detail", type=int, default=None, dest="voxel_resolution")
    args = parser.parse_args()

    source_dir = config.OUTPUT_DIR / args.source_job_id
    source_job = json.loads((source_dir / "job.json").read_text())
    raw_mesh_path = Path(source_job["raw_mesh_path"])
    if not raw_mesh_path.exists():
        raise SystemExit(f"raw mesh not found: {raw_mesh_path}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = config.OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True)

    prompt = TextPrompt(**source_job["prompt"])
    job = Job(
        job_id=job_id,
        prompt=prompt,
        target_longest_dimension_mm=args.target_longest_dimension_mm,
        status=JobStatus.RUNNING,
        stage_config={
            "smoothing_iterations": args.smoothing_iterations,
            "voxel_resolution": args.voxel_resolution,
            "rerun_from_source_job": args.source_job_id,
        },
        raw_mesh_path=raw_mesh_path,
    )

    def save() -> None:
        (job_dir / "job.json").write_text(
            json.dumps(dataclasses.asdict(job), default=_json_default, indent=2)
        )

    save()

    raw_mesh = RawMesh(mesh_path=raw_mesh_path, source_model="reused", source_format=raw_mesh_path.suffix.lstrip("."))

    print(f"Repairing at {args.target_longest_dimension_mm}mm (source: {args.source_job_id}, new job: {job_id})...")
    repair = TrimeshRepairStage()
    print_ready = repair.process(
        raw_mesh,
        args.target_longest_dimension_mm,
        job_dir,
        smoothing_iterations=args.smoothing_iterations,
        voxel_resolution=args.voxel_resolution,
    )
    job.repair_report = print_ready.repair_report
    save()

    export = STLExportStage()
    stl_result = export.export(print_ready, job_dir)
    job.stl_path = stl_result.stl_path
    job.repair_report = stl_result.repair_report
    job.status = JobStatus.SUCCEEDED
    save()

    print(f"Done: {job.stl_path}")
    r = job.repair_report
    print(f"  watertight={r.is_watertight}  bbox_mm={r.bbox_mm}")
    print(f"  below 1.2mm floor: {r.wall_thickness_below_floor_fraction:.1%}"
          f"   below nozzle width (0.4mm): {r.wall_thickness_critical_fraction:.1%}")
    for issue in r.remaining_issues:
        print(f"  [{issue.severity.value}] {issue.code}: {issue.message}")


if __name__ == "__main__":
    main()

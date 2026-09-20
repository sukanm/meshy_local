"""CLI entry point running the full pipeline end-to-end:
`uv run python -m pipeline.cli "a small dragon figurine" --scale 80`

Also the permanent debug/reproduction tool for any job — same orchestrator
the web UI (server/app.py) calls, so a CLI repro always matches what the UI
would do.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pipeline import config  # noqa: E402
from pipeline.orchestrator import Pipeline  # noqa: E402
from pipeline.stages.export.stl_export import STLExportStage  # noqa: E402
from pipeline.stages.repair.trimesh_repair import TrimeshRepairStage  # noqa: E402
from pipeline.stages.text_to_image.mflux_stage import MFluxTextToImageStage  # noqa: E402
from pipeline.types import JobStatus, TextPrompt  # noqa: E402


def build_default_pipeline(image_model: str = "trellis2") -> Pipeline:
    """image_model: "trellis2" (default — better geometric detail, faster;
    see pipeline/stages/image_to_3d/trellis2_stage.py) or "sf3d" (Stable
    Fast 3D — smoother output, texture-only detail, official MPS support)."""
    if image_model == "trellis2":
        from pipeline.stages.image_to_3d.trellis2_stage import Trellis2Stage

        image_to_3d = Trellis2Stage()
    elif image_model == "sf3d":
        from pipeline.stages.image_to_3d.sf3d_stage import StableFast3DStage

        image_to_3d = StableFast3DStage()
    else:
        raise ValueError(f"Unknown image_model: {image_model!r} (expected 'trellis2' or 'sf3d')")

    return Pipeline(
        text_to_image=MFluxTextToImageStage(),
        image_to_3d=image_to_3d,
        repair=TrimeshRepairStage(),
        export=STLExportStage(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Text prompt -> STL, end to end.")
    parser.add_argument("prompt", type=str, help="Text description of the object to generate.")
    parser.add_argument(
        "--scale",
        type=float,
        required=True,
        dest="target_longest_dimension_mm",
        help="Target longest dimension in millimeters.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducible generation.")
    parser.add_argument(
        "--image-model",
        choices=["trellis2", "sf3d"],
        default="trellis2",
        help="Image-to-3D backend (default: trellis2).",
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=None,
        dest="smoothing_iterations",
        help=f"Taubin smoothing passes after voxel-remesh repair (0=off, default: {config.VOXEL_REMESH_SMOOTHING_ITERATIONS}). Higher = smoother but less crisp detail.",
    )
    parser.add_argument(
        "--detail",
        type=int,
        default=None,
        dest="voxel_resolution",
        help=f"Voxel grid resolution for the voxel-remesh repair fallback (default: {config.VOXEL_REMESH_RESOLUTION}). Higher = finer detail preserved, slightly slower.",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        default=None,
        dest="num_candidates",
        help=f"TRELLIS.2 only: generate this many reconstructions with different seeds and keep "
        f"the cleanest by topology-defect count (default: {config.TRELLIS2_NUM_CANDIDATES}; 1 disables "
        f"this). Each extra candidate roughly multiplies image-to-3D generation time.",
    )
    args = parser.parse_args()

    print("Loading models (this can take a while on first run)...")
    pipeline = build_default_pipeline(args.image_model)

    prompt = TextPrompt(prompt=args.prompt, seed=args.seed)
    print(f"Running pipeline for: {prompt.prompt!r} (target longest dim: {args.target_longest_dimension_mm}mm)")
    job = pipeline.run(
        prompt,
        args.target_longest_dimension_mm,
        smoothing_iterations=args.smoothing_iterations,
        voxel_resolution=args.voxel_resolution,
        num_candidates=args.num_candidates,
    )

    print(f"\nJob {job.job_id}: {job.status.value}")
    if job.status == JobStatus.SUCCEEDED:
        print(f"STL: {job.stl_path}")
        r = job.repair_report
        print(f"Print-readiness: {r.status.value}")
        print(f"Watertight: {r.is_watertight}")
        print(f"Bounding box (mm): {r.bbox_mm}")
        print(f"Fits P2S build volume: {r.fits_build_volume}")
        if r.actions_taken:
            print("Actions taken:")
            for a in r.actions_taken:
                print(f"  - {a}")
        if r.remaining_issues:
            print("Issues:")
            for i in r.remaining_issues:
                print(f"  - [{i.severity.value}] {i.code}: {i.message}")
    else:
        print(f"Error: {job.error}")


if __name__ == "__main__":
    main()

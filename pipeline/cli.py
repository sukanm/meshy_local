"""CLI entry point running the full pipeline end-to-end:
`uv run python -m pipeline.cli "a small dragon figurine" --scale 80`

Also the permanent debug/reproduction tool for any job — same orchestrator
the web UI (server/app.py) calls, so a CLI repro always matches what the UI
would do.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pipeline import config  # noqa: E402
from pipeline.orchestrator import Pipeline  # noqa: E402
from pipeline.stages.export.stl_export import STLExportStage  # noqa: E402
from pipeline.stages.preview.blender_preview import BlenderPreviewStage  # noqa: E402
from pipeline.stages.print_prep.blender_print_prep import BlenderPrintPrepStage  # noqa: E402
from pipeline.stages.slice.bambu_slice import BambuSliceStage  # noqa: E402
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
        print_prep=BlenderPrintPrepStage(),
        preview=BlenderPreviewStage(),
        slicer=BambuSliceStage(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Text prompt -> STL, end to end.")
    parser.add_argument(
        "prompt",
        type=str,
        help="Text description of the object to generate (just a label when --image is given).",
    )
    parser.add_argument(
        "--image",
        default=None,
        dest="reference_image",
        help="Use this photo (a file path or an http(s) image link) instead of generating a "
        "concept image — for specific products/characters the image model doesn't know. "
        "The background is removed automatically; a single object on a plain background works best.",
    )
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
    parser.add_argument(
        "--wall-offset",
        type=float,
        default=None,
        dest="wall_offset_mm",
        help=f"Print prep: grow the whole surface outward by this many mm so every wall is at least "
        f"2x as thick (0 = off). Default: auto — {config.WALL_OFFSET_AUTO_MM}mm, applied only when walls "
        f"are likely to print as gaps. Softens fine detail and adds material.",
    )
    parser.add_argument(
        "--base",
        choices=list(config.BASE_MODES),
        default=None,
        dest="base_mode",
        help=f"Print prep: flatten the bottom for bed adhesion, add a plinth, or none "
        f"(default: {config.BASE_MODE_DEFAULT}).",
    )
    args = parser.parse_args()

    reference_image = reference_source = None
    if args.reference_image:
        from pipeline import reference_images

        try:
            if args.reference_image.startswith(("http://", "https://")):
                ref_id = reference_images.stage_image_url(args.reference_image)
            else:
                ref_id = reference_images.stage_image_file(Path(args.reference_image).expanduser())
        except reference_images.ReferenceImageError as exc:
            parser.error(str(exc))
        reference_image, reference_source = reference_images.staged_image(ref_id)
        print(f"Using reference image {args.reference_image} ({reference_source['width']}x{reference_source['height']}px)")

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
        wall_offset_mm=args.wall_offset_mm,
        base_mode=args.base_mode,
        reference_image=reference_image,
        reference_source=reference_source,
    )

    print(f"\nJob {job.job_id}: {job.status.value}")
    if job.status == JobStatus.SUCCEEDED:
        print(f"STL: {job.stl_path}")
        if job.preview_images:
            print(f"Previews: {', '.join(str(job.stl_path.parent / p) for p in job.preview_images)}")
        sr = job.slice_result
        if sr is not None:
            if sr.error:
                print(f"Bambu Studio slice: failed — {sr.error}")
            else:
                minutes = round((sr.print_time_s or 0) / 60)
                print(f"Bambu Studio project: {job.stl_path.parent / sr.project_file}")
                print(f"  {sr.settings_summary}")
                print(f"  ~{minutes // 60}h {minutes % 60}m, {sr.filament_g or 0:.1f} g filament")
                if sr.features_total:
                    print(f"  Unprinted small features: {sr.features_uncovered} of {sr.features_total}")
                for w in sr.warnings:
                    print(f"  ! {w}")
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

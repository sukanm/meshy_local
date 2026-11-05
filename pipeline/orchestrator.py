"""Wires the four pipeline stages together and persists each run as a Job
directory under output/<job_id>/ (the filesystem is the database for this
MVP — see CLAUDE.md / plan section 2).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import uuid
from pathlib import Path

import numpy as np
import trimesh

from pipeline import config
from pipeline.stages.export.base import ExportStage
from pipeline.stages.image_to_3d.base import ImageTo3DStage
from pipeline.stages.preview.base import PreviewStage
from pipeline.stages.print_prep.base import PrintPrepStage
from pipeline.stages.repair.base import RepairStage
from pipeline.stages.slice.base import SliceStage
from pipeline.stages.text_to_image.base import TextToImageStage
from pipeline.types import GeneratedImage, Job, JobStatus, TextPrompt

VIEWER_MODEL_FILENAME = "model_view.glb"
REFERENCE_IMAGE_FILENAME = "reference_image.png"
# glTF is Y-up; the final mesh is Z-up (print orientation).
_Z_UP_TO_Y_UP = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])


def write_viewer_model(stl_path: Path, job_dir: Path) -> str | None:
    """GLB copy of the final mesh for the browser viewer (model-viewer only
    reads glTF). Best-effort: the STL is the deliverable, this is a preview."""
    try:
        mesh = trimesh.load(stl_path, force="mesh")
        mesh.apply_transform(_Z_UP_TO_Y_UP)
        mesh.export(job_dir / VIEWER_MODEL_FILENAME)
        return VIEWER_MODEL_FILENAME
    except Exception:  # noqa: BLE001
        return None


def _json_default(obj):
    # numpy scalars (np.bool_, np.float64, ...) leak out of mesh math easily;
    # one in a RepairReport once made job.json unwritable, which also stopped
    # the failure from being recorded (the job stayed "RUNNING" forever).
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if hasattr(obj, "value"):  # Enum
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class Pipeline:
    """Runs text_to_image (or a supplied reference image) -> image_to_3d ->
    repair -> print_prep -> export -> preview -> slice in order. print_prep,
    preview and slicer are optional (None skips them), so a pipeline without
    Blender or Bambu Studio still works.

    Stages are injected at construction time (no plugin registry) so a
    concrete implementation can be swapped by changing one call site in
    config.py / whoever builds the Pipeline, without touching this class.
    """

    def __init__(
        self,
        text_to_image: TextToImageStage,
        image_to_3d: ImageTo3DStage,
        repair: RepairStage,
        export: ExportStage,
        print_prep: PrintPrepStage | None = None,
        preview: PreviewStage | None = None,
        slicer: SliceStage | None = None,
    ) -> None:
        self.text_to_image = text_to_image
        self.image_to_3d = image_to_3d
        self.repair = repair
        self.export = export
        self.print_prep = print_prep
        self.preview = preview
        self.slicer = slicer

    def run(
        self,
        prompt: TextPrompt,
        target_longest_dimension_mm: float,
        job_id: str | None = None,
        smoothing_iterations: int | None = None,
        voxel_resolution: int | None = None,
        num_candidates: int | None = None,
        wall_offset_mm: float | None = None,
        base_mode: str | None = None,
        reference_image: Path | None = None,
        reference_source: dict | None = None,
    ) -> Job:
        """`reference_image` (a PNG, e.g. from pipeline.reference_images)
        replaces text-to-image: it goes straight to image-to-3D, and the
        prompt is then just the job's label."""
        job = Job(
            job_id=job_id or uuid.uuid4().hex[:12],
            prompt=prompt,
            target_longest_dimension_mm=target_longest_dimension_mm,
            status=JobStatus.RUNNING,
            stage_config={
                "smoothing_iterations": smoothing_iterations,
                "voxel_resolution": voxel_resolution,
                "num_candidates": num_candidates,
                "wall_offset_mm": wall_offset_mm,
                "base_mode": base_mode,
                "reference_image": reference_source if reference_image is not None else None,
            },
        )
        job_dir = config.OUTPUT_DIR / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        self._save(job, job_dir)

        try:
            if reference_image is not None:
                job.current_stage = "reference_image"
                self._save(job, job_dir)
                image_path = job_dir / REFERENCE_IMAGE_FILENAME
                shutil.copyfile(reference_image, image_path)
                image = GeneratedImage(
                    image_path=image_path,
                    prompt=prompt,
                    generation_metadata={"source": "reference_image", **(reference_source or {})},
                )
            else:
                job.current_stage = "text_to_image"
                self._save(job, job_dir)
                image = self.text_to_image.generate(prompt, job_dir)
            job.generated_image_path = image.image_path
            self._save(job, job_dir)

            job.current_stage = "image_to_3d"
            self._save(job, job_dir)
            raw_mesh = self.image_to_3d.reconstruct(image, job_dir, num_candidates=num_candidates)
            job.raw_mesh_path = raw_mesh.mesh_path
            self._save(job, job_dir)

            job.current_stage = "repair"
            self._save(job, job_dir)
            print_ready = self.repair.process(
                raw_mesh,
                target_longest_dimension_mm,
                job_dir,
                smoothing_iterations=smoothing_iterations,
                voxel_resolution=voxel_resolution,
            )
            job.repair_report = print_ready.repair_report
            self._save(job, job_dir)

            if self.print_prep is not None:
                job.current_stage = "print_prep"
                self._save(job, job_dir)
                print_ready = self.print_prep.prepare(
                    print_ready, job_dir, wall_offset_mm=wall_offset_mm, base_mode=base_mode
                )
                job.repair_report = print_ready.repair_report
                self._save(job, job_dir)

            job.current_stage = "export"
            self._save(job, job_dir)
            stl_result = self.export.export(print_ready, job_dir)
            job.stl_path = stl_result.stl_path
            job.repair_report = stl_result.repair_report

            if self.preview is not None:
                job.current_stage = "preview"
                self._save(job, job_dir)
                job.preview_images = self.preview.render(stl_result.stl_path, job_dir)
            job.viewer_model = write_viewer_model(stl_result.stl_path, job_dir)

            if self.slicer is not None:
                job.current_stage = "slice"
                self._save(job, job_dir)
                job.slice_result = self.slicer.slice(stl_result.stl_path, job_dir)

            job.current_stage = None
            job.status = JobStatus.SUCCEEDED
        except Exception as exc:  # noqa: BLE001 - persist failure state, then re-raise
            job.status = JobStatus.FAILED
            job.error = str(exc)
            self._save(job, job_dir)
            raise
        else:
            self._save(job, job_dir)

        return job

    @staticmethod
    def _save(job: Job, job_dir: Path) -> None:
        (job_dir / "job.json").write_text(
            json.dumps(dataclasses.asdict(job), default=_json_default, indent=2)
        )

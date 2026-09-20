"""Wires the four pipeline stages together and persists each run as a Job
directory under output/<job_id>/ (the filesystem is the database for this
MVP — see CLAUDE.md / plan section 2).
"""

from __future__ import annotations

import dataclasses
import json
import uuid
from pathlib import Path

from pipeline import config
from pipeline.stages.export.base import ExportStage
from pipeline.stages.image_to_3d.base import ImageTo3DStage
from pipeline.stages.repair.base import RepairStage
from pipeline.stages.text_to_image.base import TextToImageStage
from pipeline.types import Job, JobStatus, TextPrompt


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if hasattr(obj, "value"):  # Enum
        return obj.value
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class Pipeline:
    """Runs text_to_image -> image_to_3d -> repair -> export in order.

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
    ) -> None:
        self.text_to_image = text_to_image
        self.image_to_3d = image_to_3d
        self.repair = repair
        self.export = export

    def run(
        self,
        prompt: TextPrompt,
        target_longest_dimension_mm: float,
        job_id: str | None = None,
        smoothing_iterations: int | None = None,
        voxel_resolution: int | None = None,
        num_candidates: int | None = None,
    ) -> Job:
        job = Job(
            job_id=job_id or uuid.uuid4().hex[:12],
            prompt=prompt,
            target_longest_dimension_mm=target_longest_dimension_mm,
            status=JobStatus.RUNNING,
            stage_config={
                "smoothing_iterations": smoothing_iterations,
                "voxel_resolution": voxel_resolution,
                "num_candidates": num_candidates,
            },
        )
        job_dir = config.OUTPUT_DIR / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        self._save(job, job_dir)

        try:
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

            job.current_stage = "export"
            self._save(job, job_dir)
            stl_result = self.export.export(print_ready, job_dir)
            job.stl_path = stl_result.stl_path
            job.repair_report = stl_result.repair_report
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

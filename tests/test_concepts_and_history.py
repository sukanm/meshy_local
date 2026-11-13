"""Concept images staged for review (pipeline/concepts.py) and job summaries
for the History page (pipeline/job_history.py). No models run."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image

from pipeline import concepts, job_history, reference_images
from pipeline.types import GeneratedImage, TextPrompt


class FakeTextToImage:
    def __init__(self):
        self.prompts: list[TextPrompt] = []

    def generate(self, prompt: TextPrompt, work_dir: Path) -> GeneratedImage:
        self.prompts.append(prompt)
        path = work_dir / "concept_image.png"
        Image.new("RGB", (512, 512), (240, 240, 240)).save(path)
        return GeneratedImage(path, prompt, {"model": "Fake/Model", "full_prompt": prompt.prompt + ", suffix"})


def test_make_concept_stages_image_with_its_seed(tmp_path):
    stage = FakeTextToImage()
    ref_id, seed = concepts.make_concept(stage, "an owl figurine", seed=41, directory=tmp_path)

    assert seed == 41 and stage.prompts[0].seed == 41
    path, meta = reference_images.staged_image(ref_id, tmp_path)
    assert Image.open(path).size == (512, 512)
    assert meta["kind"] == "concept"
    assert meta["seed"] == 41 and meta["prompt"] == "an owl figurine" and meta["title"] == "an owl figurine"
    assert meta["model"] == "Fake/Model"
    # The generation's temporary folder is gone; only the staged pair remains.
    assert sorted(p.suffix for p in tmp_path.iterdir()) == [".json", ".png"]


def test_make_concept_picks_a_seed_when_none_given(tmp_path):
    stage = FakeTextToImage()
    _, seed = concepts.make_concept(stage, "a robot", directory=tmp_path)
    assert isinstance(seed, int) and stage.prompts[0].seed == seed


def test_concept_request_rejects_blank_prompt():
    from pydantic import ValidationError

    from server.schemas import ConceptRequest

    with pytest.raises(ValidationError):
        ConceptRequest(prompt="   ")
    assert ConceptRequest(prompt="a cat", seed=3).seed == 3


def _write_job(output: Path, job_id: str, job: dict, files=()) -> Path:
    job_dir = output / job_id
    job_dir.mkdir(parents=True)
    for name in files:
        (job_dir / name).write_bytes(b"x")
    (job_dir / "job.json").write_text(json.dumps(job))
    return job_dir


def test_summary_of_old_job_with_absolute_paths_and_missing_fields(tmp_path):
    job_dir = _write_job(
        tmp_path,
        "old1",
        {
            "job_id": "old1",
            "status": "SUCCEEDED",
            "prompt": {"prompt": "godzilla", "seed": 42},
            "target_longest_dimension_mm": 80,
            "generated_image_path": "/Users/x/messy_local/output/old1/concept_image.png",
            "preview_images": ["/Users/x/messy_local/output/old1/preview_front.png", "preview_gone.png"],
            "stl_path": "/Users/x/messy_local/output/old1/model.stl",
            "repair_report": {"status": "PRINT_READY_WITH_WARNINGS", "wall_thickness_below_floor_fraction": 0.9},
        },
        files=["concept_image.png", "preview_front.png", "model.stl"],
    )
    s = job_history.summarize_job(json.loads((job_dir / "job.json").read_text()), job_dir)

    assert s["input_image"] == "concept_image.png"
    assert s["previews"] == ["preview_front.png"]  # missing file dropped
    assert s["thumbnail"] == "preview_front.png"
    assert s["image_source"] == "generated"
    assert s["image_model"] is None and s["text_to_image_model"] is None
    assert s["features_total"] is None and s["below_floor_fraction"] == 0.9
    assert s["base"] == "flatten"  # the default when the job didn't record one
    assert s["created_at"]  # falls back to the file's time


def test_summary_of_concept_job_reads_seed_offset_base_and_models(tmp_path):
    previews = ["preview_three_quarter.png", "preview_front.png", "preview_back.png", "preview_side.png"]
    job_dir = _write_job(
        tmp_path,
        "new1",
        {
            "job_id": "new1",
            "status": "SUCCEEDED",
            "created_at": "2026-11-06T10:00:00+00:00",
            "prompt": {"prompt": "a dragon", "seed": None},
            "stage_config": {
                "image_model": "microsoft/TRELLIS.2-4B (via trellis2mlx)",
                "num_candidates": 3,
                "base_mode": "flatten",
                "reference_image": {"kind": "concept", "seed": 12, "model": "Tongyi-MAI/Z-Image-Turbo"},
            },
            "generated_image_path": "reference_image.png",
            "preview_images": previews,
            "stl_path": "model.stl",
            "viewer_model": "model_view.glb",
            "repair_report": {
                "actions_taken": ["Thickened walls by a uniform 0.30mm outward surface offset (auto: ...)"],
                "remaining_issues": [
                    {"code": "base_skipped", "message": "Base 'flatten' couldn't be added", "severity": "warning"}
                ],
            },
            "slice_result": {"features_uncovered": 3, "features_total": 120, "print_time_s": 3600, "filament_g": 9.5},
        },
        files=["reference_image.png", "model.stl", "model_view.glb", *previews],
    )
    s = job_history.summarize_job(json.loads((job_dir / "job.json").read_text()), job_dir)

    assert s["seed"] == 12
    assert s["image_source"] == "picked concept"
    assert s["image_model"] == "TRELLIS.2"
    assert s["text_to_image_model"] == "Tongyi-MAI/Z-Image-Turbo"
    assert s["wall_offset_mm"] == 0.3
    assert s["base"] == "flatten (skipped)"
    assert s["previews"] == list(job_history.PREVIEW_ORDER)
    assert s["thumbnail"] == "preview_three_quarter.png"
    assert s["viewer_model"] == "model_view.glb"
    assert (s["features_uncovered"], s["features_total"]) == (3, 120)


@pytest.mark.parametrize(
    "value,label",
    [("trellis2", "TRELLIS.2"), ("TencentARC/Pixal3D (via pixal3d-mlx)", "Pixal3D"), ("sf3d", "Stable Fast 3D"), (None, None)],
)
def test_image_model_label(value, label):
    assert job_history.image_model_label(value) == label


def test_list_jobs_newest_first_and_skips_staging_and_broken(tmp_path):
    _write_job(tmp_path, "a", {"job_id": "a", "created_at": "2026-11-01T00:00:00+00:00"})
    _write_job(tmp_path, "b", {"job_id": "b", "created_at": "2026-11-05T00:00:00+00:00"})
    _write_job(tmp_path, "_references", {"job_id": "_references"})
    broken = tmp_path / "c"
    broken.mkdir()
    (broken / "job.json").write_text("{not json")

    assert [j["job_id"] for j in job_history.list_jobs(tmp_path)] == ["b", "a"]
    assert [j["job_id"] for j in job_history.list_jobs(tmp_path, limit=1)] == ["b"]

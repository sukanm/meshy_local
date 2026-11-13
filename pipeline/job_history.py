"""Summaries of past jobs for the web UI's History page.

Reads output/<job_id>/job.json files as they are, including old ones: jobs
from before 2026-11-04 store absolute paths (some under the old messy_local
folder), and many fields didn't exist yet. Everything here falls back to None
rather than failing, and file names are always resolved inside the job folder.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from pipeline import config

_OFFSET_RE = re.compile(r"Thickened walls by a uniform ([0-9.]+)mm")
PREVIEW_ORDER = ("preview_front.png", "preview_side.png", "preview_back.png", "preview_three_quarter.png")


def _basename(value) -> str | None:
    return Path(str(value)).name if value else None


def _existing(job_dir: Path, name: str | None) -> str | None:
    return name if name and (job_dir / name).is_file() else None


def image_model_label(value: str | None) -> str | None:
    """job.json stores either the request value ('trellis2') or the stage's
    source model ('microsoft/TRELLIS.2-4B (via trellis2mlx)')."""
    if not value:
        return None
    v = value.lower()
    if "pixal3d" in v:
        return "Pixal3D"
    if "trellis" in v:
        return "TRELLIS.2"
    if "sf3d" in v or "stable-fast" in v or "stable fast" in v:
        return "Stable Fast 3D"
    return value


def _image_source(stage_config: dict, generated_image: str | None) -> str | None:
    ref = stage_config.get("reference_image")
    if isinstance(ref, dict):
        return {
            "concept": "picked concept",
            "upload": "uploaded image",
            "web_search": "web image",
            "url": "image link",
            "file": "image file",
        }.get(ref.get("kind"), "reference image")
    return "generated" if generated_image else None


def summarize_job(job: dict, job_dir: Path) -> dict:
    stage_config = job.get("stage_config") or {}
    report = job.get("repair_report") or {}
    slice_result = job.get("slice_result") or {}
    issues = report.get("remaining_issues") or []
    actions = report.get("actions_taken") or []

    created_at = job.get("created_at")
    if not created_at:
        created_at = datetime.fromtimestamp((job_dir / "job.json").stat().st_mtime, timezone.utc).isoformat()

    input_image = _existing(job_dir, _basename(job.get("generated_image_path")))
    previews = [p for p in (_basename(n) for n in job.get("preview_images") or []) if _existing(job_dir, p)]
    previews.sort(key=lambda n: PREVIEW_ORDER.index(n) if n in PREVIEW_ORDER else len(PREVIEW_ORDER))
    thumbnail = (
        _existing(job_dir, "preview_three_quarter.png") if "preview_three_quarter.png" in previews else None
    ) or (previews[0] if previews else None) or input_image

    offset = None
    for action in actions:
        match = _OFFSET_RE.search(action)
        if match:
            offset = float(match.group(1))
    base_mode = stage_config.get("base_mode") or config.BASE_MODE_DEFAULT
    base_skipped = any(i.get("code") == "base_skipped" for i in issues)
    if not job.get("stl_path"):
        base = None
    elif base_skipped:
        base = f"{base_mode} (skipped)"
    else:
        base = base_mode

    ref = stage_config.get("reference_image")
    prompt = job.get("prompt") or {}
    seed = prompt.get("seed")
    if isinstance(ref, dict) and ref.get("kind") == "concept" and ref.get("seed") is not None:
        seed = ref["seed"]

    return {
        "job_id": job.get("job_id") or job_dir.name,
        "status": job.get("status"),
        "current_stage": job.get("current_stage"),
        "created_at": created_at,
        "prompt": prompt.get("prompt"),
        "seed": seed,
        "scale_mm": job.get("target_longest_dimension_mm"),
        "image_source": _image_source(stage_config, input_image),
        "text_to_image_model": stage_config.get("text_to_image_model")
        or (ref.get("model") if isinstance(ref, dict) else None),
        "image_model": image_model_label(stage_config.get("image_model")),
        "num_candidates": stage_config.get("num_candidates"),
        "input_image": input_image,
        "thumbnail": thumbnail,
        "previews": previews,
        "viewer_model": _existing(job_dir, job.get("viewer_model")),
        "has_stl": (job_dir / "model.stl").is_file(),
        "readiness": report.get("status"),
        "is_watertight": report.get("is_watertight"),
        "bbox_mm": report.get("bbox_mm"),
        "below_floor_fraction": report.get("wall_thickness_below_floor_fraction"),
        "below_nozzle_fraction": report.get("wall_thickness_critical_fraction"),
        "wall_offset_mm": offset,
        "base": base,
        "features_uncovered": slice_result.get("features_uncovered"),
        "features_total": slice_result.get("features_total"),
        "print_time_s": slice_result.get("print_time_s"),
        "filament_g": slice_result.get("filament_g"),
        "slice_error": slice_result.get("error"),
        "issues": [
            {"severity": i.get("severity"), "code": i.get("code"), "message": i.get("message")} for i in issues
        ],
        "error": job.get("error"),
    }


def list_jobs(output_dir: Path | None = None, limit: int = 500) -> list[dict]:
    """Newest first. Folders starting with '_' (staging) and unreadable
    job.json files are skipped."""
    output_dir = output_dir or config.OUTPUT_DIR
    jobs = []
    for job_path in output_dir.glob("*/job.json"):
        job_dir = job_path.parent
        if job_dir.name.startswith("_"):
            continue
        try:
            jobs.append(summarize_job(json.loads(job_path.read_text()), job_dir))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    jobs.sort(key=lambda j: j["created_at"] or "", reverse=True)
    return jobs[:limit]

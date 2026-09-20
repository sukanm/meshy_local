"""Thin FastAPI layer over the pipeline orchestrator. Routes contain no
pipeline logic — they only translate HTTP <-> pipeline calls — so this can
be swapped for a different frontend (e.g. a SwiftUI app calling the same
routes via a bundled sidecar) without touching pipeline/ at all.

Run with: uv run uvicorn server.app:app --host 127.0.0.1 --port 8420
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from pipeline import config  # noqa: E402
from pipeline.cli import build_default_pipeline  # noqa: E402
from pipeline.orchestrator import Pipeline  # noqa: E402
from pipeline.types import TextPrompt  # noqa: E402
from server.schemas import GenerateRequest, GenerateResponse  # noqa: E402

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="messy_local")

# Pipelines are cached per image_model, built lazily on first use rather than
# all eagerly at startup: Trellis2Stage is cheap to construct (subprocess-
# based, no eager model load), but StableFast3DStage loads its model into
# memory at construction time (~minutes on a cold Hugging Face cache) — no
# reason to pay that cost at server startup if the user never asks for it.
_pipelines: dict[str, Pipeline] = {}
_pipelines_lock = threading.Lock()
_job_lock = threading.Lock()
_job_running = False


@app.on_event("startup")
def _preload_default_pipeline() -> None:
    _get_pipeline("trellis2")


def _get_pipeline(image_model: str) -> Pipeline:
    with _pipelines_lock:
        if image_model not in _pipelines:
            print(f"Loading '{image_model}' pipeline (this can take a while on first use)...")
            _pipelines[image_model] = build_default_pipeline(image_model)
            print(f"'{image_model}' pipeline ready.")
        return _pipelines[image_model]


def _run_job(
    job_id: str,
    prompt: TextPrompt,
    scale_mm: float,
    image_model: str,
    smoothing_iterations: int | None,
    voxel_resolution: int | None,
    num_candidates: int | None,
) -> None:
    global _job_running
    try:
        # Built here, inside the background thread, not in the request
        # handler: a first-time SF3D load takes minutes, and the HTTP
        # response must return the job_id immediately regardless of model.
        pipeline = _get_pipeline(image_model)
        pipeline.run(
            prompt,
            scale_mm,
            job_id=job_id,
            smoothing_iterations=smoothing_iterations,
            voxel_resolution=voxel_resolution,
            num_candidates=num_candidates,
        )
    finally:
        with _job_lock:
            _job_running = False


@app.post("/generate")
def generate(req: GenerateRequest) -> GenerateResponse:
    global _job_running
    with _job_lock:
        if _job_running:
            raise HTTPException(
                status_code=409,
                detail="A generation job is already running (MVP is single-job-at-a-time). Wait for it to finish.",
            )
        _job_running = True

    job_id = uuid.uuid4().hex[:12]
    job_dir = config.OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "job.json").write_text(
        json.dumps(
            {
                "job_id": job_id,
                "status": "PENDING",
                "current_stage": None,
                "prompt": {"prompt": req.prompt, "seed": req.seed, "negative_prompt": None},
                "target_longest_dimension_mm": req.scale_mm,
                "stage_config": {
                    "image_model": req.image_model,
                    "smoothing_iterations": req.smoothing_iterations,
                    "voxel_resolution": req.voxel_resolution,
                    "num_candidates": req.num_candidates,
                },
            }
        )
    )

    prompt = TextPrompt(prompt=req.prompt, seed=req.seed)
    thread = threading.Thread(
        target=_run_job,
        args=(
            job_id,
            prompt,
            req.scale_mm,
            req.image_model,
            req.smoothing_iterations,
            req.voxel_resolution,
            req.num_candidates,
        ),
        daemon=True,
    )
    thread.start()

    return GenerateResponse(job_id=job_id)


@app.get("/status/{job_id}")
def status(job_id: str) -> dict:
    job_path = config.OUTPUT_DIR / job_id / "job.json"
    if not job_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return json.loads(job_path.read_text())


@app.get("/output/{job_id}/{filename}")
def get_output_file(job_id: str, filename: str) -> FileResponse:
    job_dir = (config.OUTPUT_DIR / job_id).resolve()
    path = (job_dir / filename).resolve()
    if job_dir not in path.parents or not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.get("/defaults")
def defaults() -> dict:
    """Lets the frontend render slider defaults/ranges from one source of
    truth (pipeline/config.py) instead of duplicating numbers in JS."""
    return {
        "smoothing_iterations": config.VOXEL_REMESH_SMOOTHING_ITERATIONS,
        "voxel_resolution": config.VOXEL_REMESH_RESOLUTION,
        "num_candidates": config.TRELLIS2_NUM_CANDIDATES,
    }


# Mounted last: an explicit route above always wins over this for the same
# path, and everything else falls through to serving static/index.html + app.js.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

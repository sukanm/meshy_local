"""Thin FastAPI layer over the pipeline orchestrator. Routes contain no
pipeline logic — they only translate HTTP <-> pipeline calls — so this can
be swapped for a different frontend (e.g. a SwiftUI app calling the same
routes via a bundled sidecar) without touching pipeline/ at all.

Run with: conda run -n meshy_local uvicorn server.app:app --host 127.0.0.1 --port 8420
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from pathlib import Path
from urllib.parse import unquote

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.concurrency import run_in_threadpool  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from pipeline import config, reference_images  # noqa: E402
from pipeline.cli import build_default_pipeline  # noqa: E402
from pipeline.orchestrator import Pipeline  # noqa: E402
from pipeline.types import TextPrompt  # noqa: E402
from server.schemas import (  # noqa: E402
    GenerateRequest,
    GenerateResponse,
    ReferenceImageResponse,
    ReferenceUrlRequest,
)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="meshy_local")


@app.middleware("http")
async def _revalidate_page_files(request, call_next):
    """Make browsers re-check index.html/app.js on every load. Without this a
    browser kept running an old cached app.js after a server update (it showed
    the raw model and had no .3mf button). The files still come back as a
    cheap 304 when unchanged."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-cache"
    return response

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
    wall_offset_mm: float | None,
    base_mode: str | None,
    reference_image: Path | None,
    reference_source: dict | None,
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
            wall_offset_mm=wall_offset_mm,
            base_mode=base_mode,
            reference_image=reference_image,
            reference_source=reference_source,
        )
    finally:
        with _job_lock:
            _job_running = False


@app.post("/generate")
def generate(req: GenerateRequest) -> GenerateResponse:
    global _job_running
    reference_image = reference_source = None
    if req.reference_image_id:
        try:
            reference_image, reference_source = reference_images.staged_image(req.reference_image_id)
        except reference_images.ReferenceImageError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    label = req.prompt.strip() or (reference_source or {}).get("title") or "reference image"
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
                "prompt": {"prompt": label, "seed": req.seed, "negative_prompt": None},
                "target_longest_dimension_mm": req.scale_mm,
                "stage_config": {
                    "image_model": req.image_model,
                    "smoothing_iterations": req.smoothing_iterations,
                    "voxel_resolution": req.voxel_resolution,
                    "num_candidates": req.num_candidates,
                    "wall_offset_mm": req.wall_offset_mm,
                    "base_mode": req.base_mode,
                    "reference_image": reference_source,
                },
            }
        )
    )

    prompt = TextPrompt(prompt=label, seed=req.seed)
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
            req.wall_offset_mm,
            req.base_mode,
            reference_image,
            reference_source,
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


# --- Reference images (web search, link, upload) -----------------------------
# Staged under output/_references/ and served by /output/_references/<id>.png.


def _reference_response(ref_id: str) -> ReferenceImageResponse:
    _, meta = reference_images.staged_image(ref_id)
    return ReferenceImageResponse(
        ref_id=ref_id,
        image_url=f"/output/{config.REFERENCE_IMAGE_DIR.name}/{ref_id}.png",
        width=meta["width"],
        height=meta["height"],
    )


@app.get("/image-search")
def image_search(q: str) -> dict:
    try:
        provider, results = reference_images.search_images(q[:300])
    except reference_images.ReferenceImageError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {"provider": provider, "results": reference_images.search_result_dicts(results)}


@app.post("/reference-image/from-url")
def reference_from_url(req: ReferenceUrlRequest) -> ReferenceImageResponse:
    source = {
        "kind": "web_search" if req.provider else "url",
        "provider": req.provider,
        "title": req.title,
        "page_url": req.page_url,
    }
    try:
        ref_id = reference_images.stage_image_url(
            req.url, fallback_url=req.fallback_url, source={k: v for k, v in source.items() if v}
        )
    except reference_images.ReferenceImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _reference_response(ref_id)


@app.post("/reference-image")
async def reference_upload(request: Request) -> ReferenceImageResponse:
    """Upload: the raw image bytes are the request body (no multipart, so no
    extra dependency); the original filename comes in X-Filename."""
    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > config.REFERENCE_IMAGE_MAX_BYTES:
            raise HTTPException(status_code=413, detail="That image is too large (over 25MB).")
    filename = unquote(request.headers.get("X-Filename", ""))[:200]
    try:
        ref_id = await run_in_threadpool(
            reference_images.stage_image_bytes, bytes(data), {"kind": "upload", "filename": filename}
        )
    except reference_images.ReferenceImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _reference_response(ref_id)


@app.post("/open-in-bambu/{job_id}")
def open_in_bambu(job_id: str) -> dict:
    """Opens the job's sliced project (or the STL, if slicing didn't produce
    one) in Bambu Studio on this Mac. The server and the browser run on the
    same machine in this single-user app, so launching a desktop app from a
    request is intended here."""
    job_dir = (config.OUTPUT_DIR / job_id).resolve()
    job_path = job_dir / "job.json"
    if config.OUTPUT_DIR.resolve() not in job_dir.parents or not job_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    job = json.loads(job_path.read_text())
    project = (job.get("slice_result") or {}).get("project_file")
    target = job_dir / project if project else job_dir / "model.stl"
    if not target.exists():
        raise HTTPException(status_code=404, detail="Nothing to open yet: no model.3mf or model.stl in this job")
    result = subprocess.run(["open", "-a", config.BAMBU_STUDIO_APP_NAME, str(target)], capture_output=True, text=True)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Couldn't open Bambu Studio: {result.stderr.strip() or 'unknown error'}")
    return {"opened": target.name}


@app.get("/defaults")
def defaults() -> dict:
    """Lets the frontend render slider defaults/ranges from one source of
    truth (pipeline/config.py) instead of duplicating numbers in JS."""
    return {
        "smoothing_iterations": config.VOXEL_REMESH_SMOOTHING_ITERATIONS,
        "voxel_resolution": config.VOXEL_REMESH_RESOLUTION,
        "num_candidates": config.TRELLIS2_NUM_CANDIDATES,
        "wall_offset_auto_mm": config.WALL_OFFSET_AUTO_MM,
        "base_mode": config.BASE_MODE_DEFAULT,
    }


# Mounted last: an explicit route above always wins over this for the same
# path, and everything else falls through to serving static/index.html + app.js.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

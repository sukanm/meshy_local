"""Pixal3D as an optional image-to-3D model: it runs generate_pixal3d.py in
its own env, turns the result to face the input camera, defaults to one
candidate, and is selectable from the CLI and the web API."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import trimesh
from pydantic import ValidationError

from pipeline import config
from pipeline.stages.image_to_3d import pixal3d_stage
from pipeline.stages.image_to_3d.pixal3d_stage import Pixal3DStage
from pipeline.types import GeneratedImage, TextPrompt
from server.schemas import GenerateRequest


def _stage() -> Pixal3DStage:
    # Skip __init__'s env checks: these tests replace the subprocess call.
    stage = object.__new__(Pixal3DStage)
    stage.python = Path("/nonexistent/pixal3d/python")
    stage.rembg_python = Path("/nonexistent/trellis/python3")
    stage.generate_script = Path("/nonexistent/generate_pixal3d.py")
    stage.resolution, stage.target_faces, stage.texture_size, stage.seed = 1024, 200_000, 1024, None
    return stage


def _fake_generator(calls: list):
    """Writes an asymmetric box (longest along glTF -Z) where --output points."""

    def run(cmd, **kwargs):
        calls.append(cmd)
        out = Path(cmd[cmd.index("--output") + 1])
        box = trimesh.creation.box(extents=(1.0, 2.0, 4.0))
        box.apply_translation((0, 0, -1.5))
        box.export(out)
        return subprocess.CompletedProcess(cmd, 0, stdout="MoGe MLX camera: FOV=24.0°", stderr="")

    return run


def test_generates_one_candidate_facing_the_camera(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BACKGROUND_REMOVAL_MODEL", None)
    calls: list = []
    monkeypatch.setattr(pixal3d_stage.subprocess, "run", _fake_generator(calls))
    Path(tmp_path / "in.png").touch()

    raw = _stage().reconstruct(GeneratedImage(tmp_path / "in.png", TextPrompt("x")), tmp_path)

    assert len(calls) == config.PIXAL3D_NUM_CANDIDATES == 1
    assert calls[0][0] == "/nonexistent/pixal3d/python"
    assert raw.source_model.startswith("TencentARC/Pixal3D")
    assert raw.mesh_path == tmp_path / "raw_mesh.glb"
    # Turned 180° about the vertical (glTF Y) axis: the box now extends to +Z.
    bounds = trimesh.load(raw.mesh_path, force="mesh").bounds
    np.testing.assert_allclose(bounds[:, 2], [-0.5, 3.5], atol=1e-5)
    assert "camera" in raw.generation_metadata


def test_explicit_candidates_are_honoured(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BACKGROUND_REMOVAL_MODEL", None)
    calls: list = []
    monkeypatch.setattr(pixal3d_stage.subprocess, "run", _fake_generator(calls))
    Path(tmp_path / "in.png").touch()
    stage = _stage()
    stage.seed = 7
    raw = stage.reconstruct(GeneratedImage(tmp_path / "in.png", TextPrompt("x")), tmp_path, num_candidates=2)
    assert [c[c.index("--seed") + 1] for c in calls] == ["7", "8"]
    assert raw.generation_metadata["num_candidates"] == 2


def test_missing_env_gives_a_setup_hint(monkeypatch):
    monkeypatch.setenv("MESHY_PIXAL3D_PYTHON", "/nonexistent/python")
    if not (config.VENDOR_DIR / "trellis2mlx" / ".venv" / "bin" / "python3").exists():
        pytest.skip("trellis2mlx venv not set up")
    with pytest.raises(RuntimeError, match="setup_pixal3d.sh"):
        Pixal3DStage()


def test_selectable_from_the_web_api():
    assert GenerateRequest(prompt="x", scale_mm=60, image_model="pixal3d").image_model == "pixal3d"
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="x", scale_mm=60, image_model="nope")

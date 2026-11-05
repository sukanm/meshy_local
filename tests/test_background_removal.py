"""Background removal before TRELLIS.2: the cut-out is made once per job with
config.BACKGROUND_REMOVAL_MODEL, inputs that already have transparency are
used as-is, and any failure falls back to the original image (generate.py
then removes the background itself)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pipeline import config
from pipeline.stages.image_to_3d import trellis2_stage
from pipeline.stages.image_to_3d.trellis2_stage import CUTOUT_FILENAME, Trellis2Stage, _has_transparency
from pipeline.types import GeneratedImage, TextPrompt


def _object_on_white(path: Path) -> Path:
    img = Image.new("RGB", (512, 512), "white")
    ImageDraw.Draw(img).ellipse((156, 106, 356, 406), fill=(40, 90, 160))
    img.save(path)
    return path


def _stage() -> Trellis2Stage:
    # Skip __init__'s venv check: these tests replace the subprocess call.
    stage = object.__new__(Trellis2Stage)
    stage.python = Path("/nonexistent/python3")
    return stage


def _image(path: Path) -> GeneratedImage:
    return GeneratedImage(image_path=path, prompt=TextPrompt("x"))


def test_has_transparency(tmp_path):
    Image.new("RGB", (8, 8)).save(tmp_path / "rgb.png")
    Image.new("RGBA", (8, 8), (1, 2, 3, 255)).save(tmp_path / "opaque.png")
    Image.new("RGBA", (8, 8), (1, 2, 3, 0)).save(tmp_path / "cut.png")
    assert not _has_transparency(tmp_path / "rgb.png")
    assert not _has_transparency(tmp_path / "opaque.png")
    assert _has_transparency(tmp_path / "cut.png")


def test_transparent_input_is_used_as_is(tmp_path, monkeypatch):
    Image.new("RGBA", (300, 300), (0, 0, 0, 0)).save(tmp_path / "in.png")

    def explode(*a, **k):
        raise AssertionError("background removal must be skipped for an input with transparency")

    monkeypatch.setattr(trellis2_stage.subprocess, "run", explode)
    image, info = _stage()._cut_out_background(_image(tmp_path / "in.png"), tmp_path)
    assert image.image_path == tmp_path / "in.png"
    assert "already has transparency" in info["model"]


def test_disabled_leaves_removal_to_generate_py(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BACKGROUND_REMOVAL_MODEL", None)
    src = _object_on_white(tmp_path / "in.png")
    image, info = _stage()._cut_out_background(_image(src), tmp_path)
    assert image.image_path == src and "u2net" in info["model"]


def test_failure_falls_back_to_original_image(tmp_path, monkeypatch):
    src = _object_on_white(tmp_path / "in.png")
    monkeypatch.setattr(
        trellis2_stage.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="Traceback...\nOSError: no model"),
    )
    image, info = _stage()._cut_out_background(_image(src), tmp_path)
    assert image.image_path == src
    assert info["error"].endswith("OSError: no model")


_VENV_PYTHON = trellis2_stage.TRELLIS2_DIR / ".venv" / "bin" / "python3"
_MODEL_FILE = Path.home() / ".rembg" / "models" / "birefnet-general" / "birefnet-general.onnx"


@pytest.mark.skipif(
    not (_VENV_PYTHON.exists() and _MODEL_FILE.exists()),
    reason="needs vendor/trellis2mlx/.venv and the downloaded birefnet-general model",
)
def test_real_cutout_is_rgba_with_background_removed(tmp_path):
    src = _object_on_white(tmp_path / "in.png")
    stage = _stage()
    stage.python = _VENV_PYTHON
    image, info = stage._cut_out_background(_image(src), tmp_path)
    assert image.image_path == tmp_path / CUTOUT_FILENAME
    alpha = np.asarray(Image.open(image.image_path))[:, :, 3]
    assert alpha[256, 256] > 240 and alpha[10, 10] < 15
    assert info["model"] == "birefnet-general" and 0.1 < info["foreground_fraction"] < 0.4

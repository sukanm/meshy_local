"""Reference images: decoding/normalising, staging, the public-address guard,
search-provider fallback, the thumbnail fallback, and the orchestrator using a
reference image instead of text-to-image. No network access."""

from __future__ import annotations

import io
import json

import pytest
import trimesh
from PIL import Image

from pipeline import config, reference_images
from pipeline.orchestrator import REFERENCE_IMAGE_FILENAME, Pipeline
from pipeline.reference_images import ImageSearchResult, ReferenceImageError
from pipeline.stages.export.stl_export import STLExportStage
from pipeline.stages.repair.trimesh_repair import TrimeshRepairStage
from pipeline.types import JobStatus, RawMesh, TextPrompt


def _image_bytes(size, mode="RGB", fmt="PNG", **save_kwargs) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, (200, 100, 50, 255)[: len(mode)]).save(buf, fmt, **save_kwargs)
    return buf.getvalue()


def test_normalize_downscales_and_keeps_aspect():
    img = reference_images.normalize_image(_image_bytes((2000, 1000), fmt="JPEG"))
    assert img.size == (config.REFERENCE_IMAGE_MAX_SIDE_PX, config.REFERENCE_IMAGE_MAX_SIDE_PX // 2)
    assert img.mode == "RGB"


def test_normalize_keeps_transparency():
    assert reference_images.normalize_image(_image_bytes((400, 400), mode="RGBA")).mode == "RGBA"


def test_normalize_applies_exif_orientation():
    exif = Image.Exif()
    exif[0x0112] = 6  # "rotate 90° clockwise to display"
    img = reference_images.normalize_image(_image_bytes((600, 300), fmt="JPEG", exif=exif))
    assert img.size == (300, 600)


@pytest.mark.parametrize("data", [b"not an image", _image_bytes((100, 100))])
def test_normalize_rejects_non_images_and_tiny_images(data):
    with pytest.raises(ReferenceImageError):
        reference_images.normalize_image(data)


def test_stage_and_load_round_trip(tmp_path):
    ref_id = reference_images.stage_image_bytes(_image_bytes((500, 400)), {"kind": "upload"}, tmp_path)
    assert len(ref_id) == 32
    path, meta = reference_images.staged_image(ref_id, tmp_path)
    assert path.exists() and Image.open(path).format == "PNG"
    assert meta == {"kind": "upload", "width": 500, "height": 400}


@pytest.mark.parametrize("ref_id", ["bad", "../../etc/passwd", "0" * 32])
def test_staged_image_rejects_unknown_ids(tmp_path, ref_id):
    with pytest.raises(ReferenceImageError):
        reference_images.staged_image(ref_id, tmp_path)


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/a.png",
        "file:///etc/passwd",
        "http://127.0.0.1/a.png",
        "http://localhost:8420/",
        "http://192.168.1.10/a.png",
        "http://10.0.0.1/a.png",
        "http://169.254.169.254/latest",
        "http://[::1]/a.png",
    ],
)
def test_fetch_refuses_non_public_addresses(url):
    with pytest.raises(ReferenceImageError):
        reference_images.fetch_image_bytes(url)


def _result(name="x") -> ImageSearchResult:
    return ImageSearchResult(
        title=name, image_url=f"https://e.com/{name}.png", thumbnail_url="", page_url="", width=1, height=1
    )


def test_search_falls_through_failing_and_empty_providers(monkeypatch):
    def fails(q, n):
        raise RuntimeError("rate limited")

    hit = _result()
    monkeypatch.setattr(
        reference_images,
        "_search_providers",
        lambda: [("A", fails), ("B", lambda q, n: []), ("C", lambda q, n: [hit])],
    )
    assert reference_images.search_images("godzilla") == ("C", [hit])


def test_search_reports_when_every_provider_fails(monkeypatch):
    def fails(q, n):
        raise RuntimeError("down")

    monkeypatch.setattr(reference_images, "_search_providers", lambda: [("A", fails), ("B", fails)])
    with pytest.raises(ReferenceImageError, match="A: down; B: down"):
        reference_images.search_images("godzilla")


def test_search_rejects_empty_query():
    with pytest.raises(ReferenceImageError):
        reference_images.search_images("   ")


def test_brave_is_used_first_only_with_a_key(monkeypatch):
    monkeypatch.delenv(config.BRAVE_SEARCH_API_KEY_ENV, raising=False)
    assert [n for n, _ in reference_images._search_providers()] == ["DuckDuckGo", "Openverse"]
    monkeypatch.setenv(config.BRAVE_SEARCH_API_KEY_ENV, "k")
    assert [n for n, _ in reference_images._search_providers()][0] == "Brave"


def test_stage_url_falls_back_to_thumbnail(tmp_path, monkeypatch):
    def fake_fetch(url):
        if url == "https://e.com/full.jpg":
            raise ReferenceImageError("HTTP 403")
        return _image_bytes((400, 400))

    monkeypatch.setattr(reference_images, "fetch_image_bytes", fake_fetch)
    ref_id = reference_images.stage_image_url(
        "https://e.com/full.jpg", fallback_url="https://e.com/thumb.jpg", source={"title": "t"}, directory=tmp_path
    )
    _, meta = reference_images.staged_image(ref_id, tmp_path)
    assert meta["used_thumbnail"] == "https://e.com/thumb.jpg"
    assert meta["url"] == "https://e.com/full.jpg" and meta["title"] == "t"


def test_stage_url_reports_the_original_error_when_both_fail(tmp_path, monkeypatch):
    def fake_fetch(url):
        raise ReferenceImageError(f"refused {url}")

    monkeypatch.setattr(reference_images, "fetch_image_bytes", fake_fetch)
    with pytest.raises(ReferenceImageError, match="refused https://e.com/full.jpg"):
        reference_images.stage_image_url("https://e.com/full.jpg", fallback_url="https://e.com/t.jpg", directory=tmp_path)


class _ExplodingTextToImage:
    def generate(self, prompt, job_dir):
        raise AssertionError("text-to-image must be skipped when a reference image is given")


class _RecordingImageTo3D:
    def __init__(self, mesh_path):
        self.mesh_path = mesh_path
        self.images = []

    def reconstruct(self, image, job_dir, num_candidates=None):
        self.images.append(image)
        return RawMesh(mesh_path=self.mesh_path, source_model="fake", source_format="stl")


def test_orchestrator_uses_reference_image_instead_of_text_to_image(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    stl = tmp_path / "box.stl"
    trimesh.creation.box(extents=(10, 10, 10)).export(stl)
    ref_id = reference_images.stage_image_bytes(_image_bytes((300, 300)), {"kind": "upload"}, tmp_path / "refs")
    ref_path, ref_meta = reference_images.staged_image(ref_id, tmp_path / "refs")
    image_to_3d = _RecordingImageTo3D(stl)
    pipeline = Pipeline(
        text_to_image=_ExplodingTextToImage(),
        image_to_3d=image_to_3d,
        repair=TrimeshRepairStage(),
        export=STLExportStage(),
    )
    job = pipeline.run(TextPrompt("box"), 30.0, reference_image=ref_path, reference_source=ref_meta)
    assert job.status == JobStatus.SUCCEEDED, job.error
    used = image_to_3d.images[0].image_path
    assert used.name == REFERENCE_IMAGE_FILENAME and used.read_bytes() == ref_path.read_bytes()
    saved = json.loads((used.parent / "job.json").read_text())
    assert saved["stage_config"]["reference_image"]["kind"] == "upload"
    assert saved["generated_image_path"].endswith(REFERENCE_IMAGE_FILENAME)

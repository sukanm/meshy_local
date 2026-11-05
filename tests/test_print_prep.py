"""Print-prep and preview stages (headless Blender), plus the glTF Y-up ->
Z-up conversion in the repair stage. Blender-dependent tests skip when
Blender isn't installed; the graceful-degradation path is tested regardless.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from pipeline import blender, config
from pipeline.orchestrator import Pipeline
from pipeline.stages.export.stl_export import STLExportStage
from pipeline.stages.preview.blender_preview import BlenderPreviewStage
from pipeline.stages.print_prep.blender_print_prep import BlenderPrintPrepStage, resolve_wall_offset
from pipeline.stages.repair.trimesh_repair import TrimeshRepairStage
from pipeline.types import (
    GeneratedImage,
    JobStatus,
    PrintReadinessStatus,
    RawMesh,
    RepairReport,
    TextPrompt,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
requires_blender = pytest.mark.skipif(not blender.is_available(), reason="Blender not installed")


def _repair(path: Path, scale_mm: float, work_dir: Path):
    raw = RawMesh(mesh_path=path, source_model="test", source_format=path.suffix.lstrip("."))
    return TrimeshRepairStage().process(raw, scale_mm, work_dir)


def _sphere_stl(tmp_path: Path) -> Path:
    path = tmp_path / "sphere.stl"
    trimesh.creation.icosphere(subdivisions=4, radius=10).export(path)
    return path


def test_gltf_is_converted_to_z_up(tmp_path):
    """A Y-tall glTF (Y-up by spec) comes out Z-tall, i.e. upright in a slicer."""
    path = tmp_path / "raw.glb"
    trimesh.creation.box(extents=(10, 30, 10)).export(path)
    pr = _repair(path, 60.0, tmp_path)
    extents = trimesh.load(pr.mesh_path, process=False).extents
    assert int(np.argmax(extents)) == 2
    assert extents[2] == pytest.approx(60.0, abs=0.5)
    assert any("Z-up" in a for a in pr.repair_report.actions_taken)


def test_stl_input_is_not_rotated(tmp_path):
    """STL is already Z-up; only glTF input gets the axis conversion."""
    path = tmp_path / "raw.stl"
    trimesh.creation.box(extents=(10, 30, 10)).export(path)
    pr = _repair(path, 60.0, tmp_path)
    extents = trimesh.load(pr.mesh_path, process=False).extents
    assert int(np.argmax(extents)) == 1
    assert not any("Z-up" in a for a in pr.repair_report.actions_taken)


@pytest.mark.parametrize(
    ("requested", "critical", "expected"),
    [
        (0, 0.9, (0.0, "off")),
        (0.5, 0.0, (0.5, "requested")),
        (None, 0.5, (config.WALL_OFFSET_AUTO_MM, "auto")),
        (None, 0.0, (0.0, "auto-not-needed")),
        (None, None, (0.0, "auto-not-needed")),
    ],
)
def test_resolve_wall_offset_modes(requested, critical, expected):
    """None = auto (triggered by the below-nozzle fraction), 0 = off, >0 = forced."""
    report = RepairReport(is_watertight=True, was_repaired=False, wall_thickness_critical_fraction=critical)
    assert resolve_wall_offset(requested, report) == expected


@requires_blender
def test_offset_thickens_thin_plate(tmp_path):
    """A 0.3mm plate (below one nozzle width) offset by 0.6mm comes out ~1.5mm thick, closed, with the
    thin-wall warning cleared."""
    plate = tmp_path / "plate.stl"
    trimesh.creation.box(extents=(20, 20, 0.3)).export(plate)
    pr = _repair(plate, 20.0, tmp_path)
    assert pr.repair_report.wall_thickness_critical_fraction > 0.5  # really thin before
    out = BlenderPrintPrepStage().prepare(pr, tmp_path, wall_offset_mm=0.6, base_mode="none")
    r = out.repair_report
    m = trimesh.load(out.mesh_path, process=False)
    assert r.is_watertight
    assert m.extents[2] == pytest.approx(0.3 + 2 * 0.6, abs=0.1)
    # Not min_wall_thickness_mm: the estimator samples random points, and a ray grazing the
    # offset plate's rounded rim occasionally reads ~0.8mm, which made this test flaky.
    assert r.wall_thickness_critical_fraction == 0
    assert r.wall_thickness_below_floor_fraction < 0.05
    assert not any(i.code == "thin_walls" for i in r.remaining_issues)
    assert any("Thickened walls" in a for a in r.actions_taken)


@requires_blender
def test_offset_fills_sealed_cavity_and_says_so(tmp_path):
    """OpenVDB's flood-fill sign treats a sealed cavity as interior, so a
    hollow box comes out solid — must be disclosed, not silent."""
    pr = _repair(FIXTURES_DIR / "thin_wall.stl", 40.0, tmp_path)
    out = BlenderPrintPrepStage().prepare(pr, tmp_path, wall_offset_mm=0.6, base_mode="none")
    m = trimesh.load(out.mesh_path, process=False)
    assert out.repair_report.is_watertight
    assert m.volume == pytest.approx(41.2**3, rel=0.02)
    assert any("Filled 1 sealed internal cavity" in a for a in out.repair_report.actions_taken)


@requires_blender
def test_prepared_report_serializes_to_job_json(tmp_path):
    """Regression: a numpy.bool_ in the report made job.json unwritable and
    left a real web-UI job stuck at "RUNNING print_prep"."""
    import dataclasses
    import json

    from pipeline.orchestrator import _json_default

    pr = _repair(_sphere_stl(tmp_path), 20.0, tmp_path)
    out = BlenderPrintPrepStage().prepare(pr, tmp_path, wall_offset_mm=0.6, base_mode="flatten")
    assert type(out.repair_report.is_watertight) is bool
    assert type(out.repair_report.fits_build_volume) is bool
    json.dumps(dataclasses.asdict(out.repair_report), default=_json_default)


@requires_blender
def test_flatten_base_creates_flat_bottom(tmp_path):
    """Flatten leaves a downward-facing flat cap resting exactly on Z=0."""
    pr = _repair(_sphere_stl(tmp_path), 20.0, tmp_path)
    out = BlenderPrintPrepStage().prepare(pr, tmp_path, wall_offset_mm=0, base_mode="flatten")
    m = trimesh.load(out.mesh_path, process=False)
    assert m.bounds[0][2] == pytest.approx(0.0, abs=1e-6)
    flat_bottom = (m.face_normals[:, 2] < -0.99) & (m.triangles_center[:, 2] < 1e-3)
    assert flat_bottom.sum() >= 10
    assert out.repair_report.is_watertight
    assert m.extents[2] == pytest.approx(20.0 - config.FLATTEN_BASE_DEPTH_MM, abs=0.05)


@requires_blender
def test_plinth_base_grows_footprint(tmp_path):
    """Plinth extends past the model's XY footprint by the configured margin."""
    pr = _repair(_sphere_stl(tmp_path), 20.0, tmp_path)
    sphere_xy = trimesh.load(pr.mesh_path, process=False).extents[:2]
    out = BlenderPrintPrepStage().prepare(pr, tmp_path, wall_offset_mm=0, base_mode="plinth")
    m = trimesh.load(out.mesh_path, process=False)
    assert (m.extents[:2] >= sphere_xy + 2 * config.PLINTH_MARGIN_MM - 0.5).all()
    assert out.repair_report.is_watertight
    assert out.repair_report.bbox_mm[2] == pytest.approx(m.extents[2])


def test_blender_missing_degrades_gracefully(tmp_path, monkeypatch):
    """No Blender: the job still produces a mesh, with skip warnings, not an exception."""
    monkeypatch.setenv("MESHY_BLENDER", "/nonexistent/blender")
    monkeypatch.setattr(config, "BLENDER_EXECUTABLE", "/nonexistent/blender")
    monkeypatch.setattr("pipeline.blender.shutil.which", lambda *_: None)

    pr = _repair(FIXTURES_DIR / "thin_wall.stl", 40.0, tmp_path)
    out = BlenderPrintPrepStage().prepare(pr, tmp_path, wall_offset_mm=0.6, base_mode="flatten")
    codes = {i.code for i in out.repair_report.remaining_issues}
    assert {"wall_offset_skipped", "base_skipped"} <= codes
    assert out.mesh_path.exists()
    assert out.repair_report.status != PrintReadinessStatus.NEEDS_MANUAL_REPAIR
    assert BlenderPreviewStage().render(_sphere_stl(tmp_path), tmp_path) == []


@requires_blender
def test_preview_renders_pngs(tmp_path):
    """Every configured view is rendered as a real PNG."""
    stl = tmp_path / "model.stl"
    trimesh.creation.box(extents=(10, 20, 30)).export(stl)
    names = BlenderPreviewStage().render(stl, tmp_path)
    assert len(names) == len(config.PREVIEW_VIEWS)
    for name in names:
        assert (tmp_path / name).read_bytes()[:4] == b"\x89PNG"


class _FakeTextToImage:
    def generate(self, prompt, job_dir):
        return GeneratedImage(image_path=job_dir / "x.png", prompt=prompt)


class _FakeImageTo3D:
    def __init__(self, mesh_path: Path):
        self.mesh_path = mesh_path

    def reconstruct(self, image, job_dir, num_candidates=None):
        return RawMesh(mesh_path=self.mesh_path, source_model="fake", source_format="stl")


class _RecordingPrintPrep:
    def __init__(self):
        self.calls = []

    def prepare(self, mesh, work_dir, wall_offset_mm=None, base_mode=None):
        self.calls.append((wall_offset_mm, base_mode))
        return mesh


class _FakePreview:
    def render(self, stl_path, work_dir):
        return ["preview_front.png"]


class _FakeSlicer:
    def __init__(self):
        self.stl_paths = []

    def slice(self, stl_path, work_dir):
        from pipeline.types import SliceResult

        self.stl_paths.append(stl_path)
        return SliceResult(project_file="model.3mf", print_time_s=60.0)


def test_orchestrator_runs_optional_stages(tmp_path, monkeypatch):
    """Print prep receives the job's options; preview filenames, the viewer
    GLB and the slice result land on the Job."""
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    stl = tmp_path / "box.stl"
    trimesh.creation.box(extents=(10, 10, 10)).export(stl)
    print_prep = _RecordingPrintPrep()
    pipeline = Pipeline(
        text_to_image=_FakeTextToImage(),
        image_to_3d=_FakeImageTo3D(stl),
        repair=TrimeshRepairStage(),
        export=STLExportStage(),
        print_prep=print_prep,
        preview=_FakePreview(),
        slicer=(slicer := _FakeSlicer()),
    )
    job = pipeline.run(TextPrompt("x"), 30.0, wall_offset_mm=0.3, base_mode="none")
    assert job.status == JobStatus.SUCCEEDED
    assert print_prep.calls == [(0.3, "none")]
    assert job.preview_images == ["preview_front.png"]
    assert slicer.stl_paths == [job.stl_path]
    assert job.slice_result.project_file == "model.3mf"
    # Viewer GLB is the final mesh turned back to glTF's Y-up: Z extent -> Y.
    glb = trimesh.load(job.stl_path.parent / job.viewer_model, force="mesh")
    stl = trimesh.load(job.stl_path)
    assert glb.extents[1] == pytest.approx(stl.extents[2], rel=1e-4)
    saved = json.loads((job.stl_path.parent / "job.json").read_text())
    assert saved["slice_result"]["project_file"] == "model.3mf"


def test_voxel_fill_is_solid_despite_surface_holes():
    """Regression: a leaky surface (every raw TRELLIS.2 mesh) used to voxel-fill
    as a hollow double skin, because `VoxelGrid.fill()` only fills cavities the
    surface fully encloses. The interior must come out solid."""
    from pipeline.stages.repair.trimesh_repair import _solid_fill

    sphere = trimesh.creation.icosphere(subdivisions=5, radius=10)
    rng = np.random.default_rng(0)
    centers = sphere.vertices[rng.choice(len(sphere.vertices), 40, replace=False)]
    dist = np.linalg.norm(sphere.triangles_center[:, None] - centers[None], axis=2).min(axis=1)
    leaky = sphere.copy()
    leaky.update_faces(dist > 0.6)  # 40 holes ~1.2mm across, like TRELLIS output
    leaky.remove_unreferenced_vertices()
    assert not leaky.is_watertight
    pitch = 0.4
    surface = leaky.voxelized(pitch=pitch)
    assert surface.copy().fill().matrix.sum() * pitch**3 < 0.25 * sphere.volume  # the old fill stays hollow
    solid = _solid_fill(surface.matrix, config.VOXEL_FILL_SEAL_VOXELS)
    assert solid.sum() * pitch**3 == pytest.approx(sphere.volume, rel=0.15)

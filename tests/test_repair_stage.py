"""Phase 3 verification: the repair stage exercised against synthetic broken
meshes, independent of whether the generative stages (Phase 1/2) work at
all. See pipeline/stages/repair/trimesh_repair.py for the escalation ladder
this is testing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh

from pipeline.stages.export.stl_export import STLExportStage
from pipeline.stages.repair.trimesh_repair import TrimeshRepairStage
from pipeline.types import PrintReadinessStatus, RawMesh, Severity


def _assert_strictly_manifold(mesh_path: Path) -> None:
    """Direct edge-multiplicity check on a freshly-loaded mesh, independent
    of trimesh's cached `is_watertight` property. Added after a real bug
    during development: `.is_watertight` can read a stale cached value after
    in-place vertex mutation (e.g. the detail-recovery snap in
    _voxel_remesh), which would silently hide a real defect. This check
    can't be fooled the same way since it re-derives edges from scratch.

    Always welds vertices by exact position first: a safe no-op for
    already-indexed formats (.ply) where coincident positions already share
    an index, and necessary for .stl, which stores each triangle's vertices
    independently with no shared indexing at all — every edge would
    otherwise appear as a false "boundary" edge regardless of the mesh's
    real topology (a real STL consumer, e.g. a slicer, must weld the same
    way to determine manifoldness at all)."""
    m = trimesh.load(mesh_path, force="mesh", process=False)
    m.merge_vertices(digits_vertex=8)
    edges_sorted = np.sort(m.edges, axis=1)
    _unique, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    assert (counts == 2).all(), (
        f"{mesh_path}: {((counts == 1).sum())} boundary edges, "
        f"{(counts > 2).sum()} non-manifold edges"
    )

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def _raw_mesh(filename: str) -> RawMesh:
    return RawMesh(
        mesh_path=FIXTURES_DIR / filename,
        source_model="synthetic-fixture",
        source_format="stl",
    )


@pytest.fixture
def stage() -> TrimeshRepairStage:
    return TrimeshRepairStage()


def test_hole_gets_filled_and_becomes_watertight(stage, tmp_path):
    result = stage.process(_raw_mesh("hole_cube.stl"), target_longest_dimension_mm=50.0, work_dir=tmp_path)
    report = result.repair_report
    assert report.is_watertight
    assert report.status == PrintReadinessStatus.PRINT_READY
    assert not report.remaining_issues


def test_flipped_normals_gets_fixed(stage, tmp_path):
    result = stage.process(
        _raw_mesh("flipped_normals_cube.stl"), target_longest_dimension_mm=50.0, work_dir=tmp_path
    )
    report = result.repair_report
    assert report.is_watertight
    assert report.status == PrintReadinessStatus.PRINT_READY


def test_multi_shell_keeps_largest_and_discloses_removal(stage, tmp_path):
    result = stage.process(_raw_mesh("multi_shell.stl"), target_longest_dimension_mm=50.0, work_dir=tmp_path)
    report = result.repair_report
    assert report.is_watertight
    assert any("disconnected fragment" in action for action in report.actions_taken)
    # Disclosed, not blocking: the small floater is discarded automatically.
    assert report.status == PrintReadinessStatus.PRINT_READY


def test_thin_wall_flags_warning_without_auto_fixing(stage, tmp_path):
    # 20-unit box with a 0.5-unit wall each side; scaling longest dim to
    # 20mm keeps the wall at ~0.5mm, below the 1.2mm FDM floor.
    result = stage.process(_raw_mesh("thin_wall.stl"), target_longest_dimension_mm=20.0, work_dir=tmp_path)
    report = result.repair_report
    assert report.is_watertight
    thin_wall_issues = [i for i in report.remaining_issues if i.code == "thin_walls"]
    assert len(thin_wall_issues) == 1
    assert thin_wall_issues[0].severity == Severity.WARNING
    assert report.status == PrintReadinessStatus.PRINT_READY_WITH_WARNINGS
    # Never auto-fixed: geometry should not have been scaled up to compensate.
    assert report.min_wall_thickness_mm is not None
    assert report.min_wall_thickness_mm < 1.2


def test_pymeshlab_failure_falls_through_to_voxel_remesh(stage, tmp_path, monkeypatch):
    # Regression test: PyMeshLab's own filters (e.g.
    # meshing_re_orient_faces_coherently) can require the mesh already be
    # manifold and raise rather than degrade gracefully. Observed on real
    # TRELLIS.2 output during development — verify the stage falls through
    # to voxel remeshing instead of crashing.
    import pipeline.stages.repair.trimesh_repair as repair_module

    monkeypatch.setattr(repair_module, "_baseline_repair", lambda mesh: None)

    def _raise(mesh):
        raise RuntimeError("simulated PyMeshLab failure on non-manifold input")

    monkeypatch.setattr(repair_module, "_pymeshlab_repair", _raise)

    result = stage.process(_raw_mesh("hole_cube.stl"), target_longest_dimension_mm=50.0, work_dir=tmp_path)
    report = result.repair_report
    assert report.is_watertight
    assert any("PyMeshLab repair failed" in action for action in report.actions_taken)
    assert any("voxel remeshing" in action.lower() for action in report.actions_taken)
    _assert_strictly_manifold(result.mesh_path)


def test_voxel_remesh_output_survives_stl_export(stage, tmp_path, monkeypatch):
    # Regression test for a real bug: STLExportStage used to reload the
    # repaired mesh with trimesh's default on-load processing (vertex
    # merging/dedup), which silently reintroduced ~1.6M non-manifold edges
    # into a mesh that the repair stage's detail-recovery snap had made
    # perfectly valid — then re-exported that damaged version as the final
    # STL. Verify the actual .stl the pipeline hands to the user is
    # strictly manifold, not just the intermediate .ply the repair stage
    # itself wrote.
    import pipeline.stages.repair.trimesh_repair as repair_module

    monkeypatch.setattr(repair_module, "_baseline_repair", lambda mesh: None)

    def _raise(mesh):
        raise RuntimeError("simulated PyMeshLab failure on non-manifold input")

    monkeypatch.setattr(repair_module, "_pymeshlab_repair", _raise)

    print_ready = stage.process(_raw_mesh("hole_cube.stl"), target_longest_dimension_mm=50.0, work_dir=tmp_path)
    result = STLExportStage().export(print_ready, work_dir=tmp_path)
    _assert_strictly_manifold(result.stl_path)


def test_exceeds_build_volume_is_blocking(stage, tmp_path):
    # Requesting a 300mm part exceeds the P2S's 256mm build volume.
    result = stage.process(_raw_mesh("hole_cube.stl"), target_longest_dimension_mm=300.0, work_dir=tmp_path)
    report = result.repair_report
    assert report.fits_build_volume is False
    blocking = [i for i in report.remaining_issues if i.code == "exceeds_build_volume"]
    assert len(blocking) == 1
    assert blocking[0].severity == Severity.BLOCKING
    assert report.status == PrintReadinessStatus.NEEDS_MANUAL_REPAIR

"""Bambu Studio slice stage: profile flattening, G-code parsing, the
unprinted-feature check, graceful degradation, and (when Bambu Studio is
installed) a real slice."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import trimesh

from pipeline import config
from pipeline.stages.slice import bambu_slice
from pipeline.stages.slice.bambu_slice import (
    BambuSliceStage,
    flatten_profile,
    parse_gcode_extrusions,
    unprinted_features,
)

requires_bambu = pytest.mark.skipif(not bambu_slice.is_available(), reason="Bambu Studio not installed")


def test_flatten_profile_resolves_inherits_chain(tmp_path):
    (tmp_path / "process").mkdir()
    (tmp_path / "process" / "base.json").write_text(json.dumps({"layer_height": "0.2", "wall_loops": "2"}))
    (tmp_path / "process" / "mid.json").write_text(json.dumps({"inherits": "base", "layer_height": "0.12"}))
    (tmp_path / "process" / "leaf.json").write_text(json.dumps({"inherits": "mid", "name": "leaf"}))
    flat = flatten_profile("process", "leaf", tmp_path)
    assert flat == {"layer_height": "0.12", "wall_loops": "2", "name": "leaf"}


def test_parse_gcode_keeps_model_extrusion_only(tmp_path):
    gcode = tmp_path / "plate_1.gcode"
    gcode.write_text(
        "\n".join(
            [
                "M83",
                "; Z_HEIGHT: 0.2",
                "; FEATURE: Outer wall",
                "G1 X0 Y0",
                "G1 X10 Y0 E0.5",  # model extrusion
                "G1 X10 Y10",  # travel
                "; FEATURE: Support",
                "G1 X20 Y10 E0.5",  # support: excluded
                "; Z_HEIGHT: 0.4",
                "; FEATURE: Inner wall",
                "G1 X20 Y20 E0.3 ; comment",
                "G1 X20 Y25 E-0.8",  # retraction
            ]
        )
    )
    layers = parse_gcode_extrusions(gcode)
    assert layers[0.2] == [(0.0, 0.0, 10.0, 0.0)]
    assert layers[0.4] == [(20.0, 10.0, 20.0, 20.0)]


def test_unprinted_features_counts_features_without_plastic():
    """Two 4mm-square posts; extrusion only traces the left one."""
    left = trimesh.creation.box(extents=(4, 4, 3))
    left.apply_translation([2, 2, 1.5])
    right = left.copy()
    right.apply_translation([10, 0, 0])
    mesh = trimesh.util.concatenate([left, right])
    square = [(0.2, 0.2, 3.8, 0.2), (3.8, 0.2, 3.8, 3.8), (3.8, 3.8, 0.2, 3.8), (0.2, 3.8, 0.2, 0.2)]
    layers = {round(0.2 * i, 2): list(square) for i in range(1, 15)}  # 0.2mm layers up to 2.8mm
    total, uncovered = unprinted_features(mesh, layers, line_width_mm=0.42)
    # Sampled every 1mm from the first layer (0.2, 1.2, 2.2): two posts each time.
    assert (total, uncovered) == (6, 3)


def test_missing_bambu_studio_reports_error_not_exception(tmp_path, monkeypatch):
    monkeypatch.setenv("MESHY_BAMBU_STUDIO", "/nonexistent/BambuStudio")
    monkeypatch.setattr(config, "BAMBU_STUDIO_EXECUTABLE", "/nonexistent/BambuStudio")
    monkeypatch.setattr("pipeline.stages.slice.bambu_slice.shutil.which", lambda *_: None)
    stl = tmp_path / "model.stl"
    trimesh.creation.box(extents=(10, 10, 10)).export(stl)
    result = BambuSliceStage().slice(stl, tmp_path)
    assert result.error and "not found" in result.error
    assert result.project_file is None


@requires_bambu
def test_real_slice_produces_project_and_estimates(tmp_path):
    stl = tmp_path / "model.stl"
    box = trimesh.creation.box(extents=(20, 20, 10))
    box.apply_translation([0, 0, 5])
    box.export(stl)
    result = BambuSliceStage().slice(stl, tmp_path)
    assert result.error is None, result.error
    assert (tmp_path / result.project_file).exists()
    assert result.print_time_s > 0 and result.filament_g > 0
    assert result.layer_height_mm == pytest.approx(0.12)
    assert "Arachne" in result.settings_summary
    assert result.features_total > 0 and result.features_uncovered == 0
    # The temporary slicing directory is cleaned up.
    assert not [p for p in tmp_path.iterdir() if p.name.startswith("slice_")]

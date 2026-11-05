"""Slice stage backed by the Bambu Studio CLI (added 2026-11-04).

Runs after export on the final STL and produces:
  - `model.3mf`: a sliced Bambu Studio project (P2S, PLA, the settings in
    config.BAMBU_PROCESS_OVERRIDES) the user can open and print directly;
  - the slicer's own print-time and filament estimate;
  - an unprinted-feature check: which parts of the model the G-code actually
    puts no plastic in. This is ground truth from the real slicer, not our
    ray-cast wall-thickness heuristic — it's what exposed the hollow
    double-skin bug (CLAUDE.md, 2026-11-04).

Two CLI gotchas this works around:
  - The CLI doesn't resolve system profiles' `inherits` chains (it silently
    falls back to defaults, e.g. 0.2mm layers), so each chain is flattened
    into one JSON here first.
  - With `--arrange 1` the CLI may move or rotate the model, which would
    misalign the G-code with the mesh for the coverage check; the mesh is
    placed at the plate centre instead and arrange is disabled.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from pipeline import config
from pipeline.types import SliceResult

PROJECT_FILENAME = "model.3mf"

# G-code feature types that aren't part of the model itself.
_NON_MODEL_FEATURES = ("support", "skirt", "brim", "prime", "wipe", "purge", "custom")


class BambuStudioUnavailableError(RuntimeError):
    pass


def find_bambu_studio() -> str:
    for candidate in (os.environ.get("MESHY_BAMBU_STUDIO"), config.BAMBU_STUDIO_EXECUTABLE, shutil.which("bambu-studio")):
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise BambuStudioUnavailableError(
        "Bambu Studio not found (install it, or set MESHY_BAMBU_STUDIO to its executable)"
    )


def is_available() -> bool:
    try:
        find_bambu_studio()
    except BambuStudioUnavailableError:
        return False
    return config.BAMBU_PROFILE_DIR.is_dir()


def flatten_profile(kind: str, name: str, profile_dir: Path | None = None) -> dict:
    """Resolve a system profile's `inherits` chain into one flat dict."""
    profile_dir = profile_dir or config.BAMBU_PROFILE_DIR
    path = profile_dir / kind / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Bambu Studio profile not found: {path}")
    data = json.loads(path.read_text())
    parent = data.pop("inherits", None)
    if parent:
        merged = flatten_profile(kind, parent, profile_dir)
        merged.update(data)
        data = merged
    return data


def _write_profiles(tmp: Path) -> tuple[Path, Path, Path, dict]:
    machine = flatten_profile("machine", config.BAMBU_MACHINE_PROFILE)
    process = flatten_profile("process", config.BAMBU_PROCESS_PROFILE)
    filament = flatten_profile("filament", config.BAMBU_FILAMENT_PROFILE)
    process.update(config.BAMBU_PROCESS_OVERRIDES)
    # Flattened process/filament presets are no longer system presets.
    process["from"] = "User"
    filament["from"] = "User"
    paths = []
    for label, data in (("machine", machine), ("process", process), ("filament", filament)):
        p = tmp / f"{label}.json"
        p.write_text(json.dumps(data))
        paths.append(p)
    return paths[0], paths[1], paths[2], {"machine": machine, "process": process, "filament": filament}


def _settings_summary(process: dict, filament: dict) -> str:
    walls = "Arachne walls" if process.get("wall_generator") == "arachne" else "classic walls"
    if process.get("wall_generator") == "arachne" and "min_feature_size" in process:
        walls += f" (fine features down to {process['min_feature_size']} of nozzle)"
    supports = (
        f"{process.get('support_type', 'auto')} supports" if str(process.get("enable_support")) == "1" else "no supports"
    )
    return f"{config.BAMBU_PROCESS_PROFILE} · {config.BAMBU_FILAMENT_PROFILE} · {walls} · {supports}"


def parse_gcode_extrusions(gcode_path: Path) -> dict[float, list[tuple[float, float, float, float]]]:
    """Model-feature extrusion segments per layer: {layer_top_z: [(x0, y0, x1, y1), ...]}.
    Arcs (G2/G3) are approximated by their chord — fine for a coverage check."""
    layers: dict[float, list] = {}
    current = None
    x = y = None
    model_feature = True
    absolute_e = False
    last_e = 0.0
    move = re.compile(r"^G[0-3]\b")
    for line in gcode_path.open():
        if line.startswith(";"):
            if line.startswith("; Z_HEIGHT:"):
                current = layers.setdefault(round(float(line.split(":", 1)[1]), 4), [])
            elif line.startswith("; FEATURE:"):
                feature = line.split(":", 1)[1].strip().lower()
                model_feature = not any(word in feature for word in _NON_MODEL_FEATURES)
            continue
        if line.startswith("M82"):
            absolute_e = True
        elif line.startswith("M83"):
            absolute_e = False
        elif line.startswith("G92") and " E" in line:
            last_e = float(re.search(r"E([-\d.]+)", line).group(1))
        if not move.match(line):
            continue
        nx, ny, e = x, y, None
        for token in line.split(";", 1)[0].split()[1:]:
            axis, value = token[0], token[1:]
            if axis == "X":
                nx = float(value)
            elif axis == "Y":
                ny = float(value)
            elif axis == "E":
                e = float(value)
        extruded = False
        if e is not None:
            delta = e - last_e if absolute_e else e
            if absolute_e:
                last_e = e
            extruded = delta > 0
        if extruded and model_feature and current is not None and x is not None and (nx, ny) != (x, y):
            current.append((x, y, nx, ny))
        x, y = nx, ny
    return layers


def _section_mask(mesh: trimesh.Trimesh, z: float, origin: np.ndarray, shape: tuple[int, int], px: float) -> np.ndarray:
    from PIL import Image, ImageDraw

    mask = np.zeros(shape, dtype=bool)
    section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    if section is None:
        return mask
    for loop in section.discrete:
        pts = (np.asarray(loop)[:, :2] - origin) / px
        if len(pts) < 3:
            continue
        im = Image.new("1", (shape[1], shape[0]))
        ImageDraw.Draw(im).polygon([tuple(p) for p in pts], fill=1)
        mask ^= np.array(im, dtype=bool)  # even-odd: holes cancel out
    return mask


def _path_mask(segments, origin: np.ndarray, shape: tuple[int, int], px: float, width_mm: float) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.new("1", (shape[1], shape[0]))
    draw = ImageDraw.Draw(im)
    width = max(1, int(round(width_mm / px)))
    for x0, y0, x1, y1 in segments:
        draw.line([((x0 - origin[0]) / px, (y0 - origin[1]) / px), ((x1 - origin[0]) / px, (y1 - origin[1]) / px)], fill=1, width=width)
    return np.array(im, dtype=bool)


def unprinted_features(
    mesh: trimesh.Trimesh,
    layers: dict[float, list],
    line_width_mm: float,
) -> tuple[int, int]:
    """(features_total, features_uncovered): connected cross-section regions
    sampled every config.SLICE_CHECK_STEP_MM of height, and how many get
    (almost) no extrusion. `mesh` must be in the same plate coordinates as
    the G-code."""
    from scipy import ndimage

    zs = np.array(sorted(layers))
    if len(zs) < 2:
        return 0, 0
    layer_h = float(np.median(np.diff(zs)))
    px = config.SLICE_CHECK_PIXEL_MM
    origin = mesh.bounds[0, :2] - 1.0
    span = mesh.bounds[1, :2] - mesh.bounds[0, :2] + 2.0
    shape = (int(np.ceil(span[1] / px)), int(np.ceil(span[0] / px)))
    min_px = config.SLICE_CHECK_MIN_FEATURE_MM2 / (px * px)

    total = uncovered = 0
    for target in np.arange(zs[0], zs[-1], config.SLICE_CHECK_STEP_MM):
        z = float(zs[np.argmin(np.abs(zs - target))])
        section = _section_mask(mesh, z - layer_h / 2, origin, shape, px)
        if not section.any():
            continue
        paths = _path_mask(layers[z], origin, shape, px, line_width_mm)
        labels, n = ndimage.label(section)
        if n == 0:
            continue
        index = np.arange(1, n + 1)
        area = ndimage.sum(section, labels, index)
        covered = ndimage.sum(section & paths, labels, index)
        keep = area >= min_px
        total += int(keep.sum())
        uncovered += int(((covered[keep] / area[keep]) < config.SLICE_CHECK_COVERED_FRACTION).sum())
    return total, uncovered


class BambuSliceStage:
    def slice(self, stl_path: Path, work_dir: Path) -> SliceResult:
        try:
            exe = find_bambu_studio()
        except BambuStudioUnavailableError as exc:
            return SliceResult(error=str(exc))
        try:
            with tempfile.TemporaryDirectory(prefix="slice_", dir=work_dir) as tmp_name:
                return self._slice(exe, stl_path, work_dir, Path(tmp_name))
        except Exception as exc:  # noqa: BLE001 - best-effort stage, report instead of failing the job
            return SliceResult(error=f"{type(exc).__name__}: {exc}")

    def _slice(self, exe: str, stl_path: Path, work_dir: Path, tmp: Path) -> SliceResult:
        machine_path, process_path, filament_path, profiles = _write_profiles(tmp)
        summary = _settings_summary(profiles["process"], profiles["filament"])

        # Place at the plate centre, resting on the bed (see module docstring).
        mesh = trimesh.load(stl_path, force="mesh")
        plate = np.array([[float(v) for v in xy.split("x")] for xy in profiles["machine"]["printable_area"]])
        centre = (plate.min(axis=0) + plate.max(axis=0)) / 2
        mesh.apply_translation([*(centre - (mesh.bounds[0, :2] + mesh.bounds[1, :2]) / 2), -mesh.bounds[0, 2]])
        placed = tmp / "model.stl"
        mesh.export(placed)

        cmd = [
            exe,
            "--orient", "0",
            "--arrange", "0",
            "--load-settings", f"{machine_path};{process_path}",
            "--load-filaments", str(filament_path),
            "--slice", "0",
            "--outputdir", str(tmp),
            "--export-3mf", PROJECT_FILENAME,
            str(placed),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=config.BAMBU_SLICE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return SliceResult(settings_summary=summary, error=f"Bambu Studio timed out after {config.BAMBU_SLICE_TIMEOUT_S}s")

        result_path = tmp / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        gcode = tmp / "plate_1.gcode"
        if proc.returncode != 0 or result.get("return_code", 0) != 0 or not gcode.exists():
            message = result.get("error_string") or (proc.stdout + proc.stderr).strip().splitlines()[-1:] or ["unknown error"]
            reasons = [line.split("slicing warnings:", 1)[1].strip() for line in (proc.stdout + proc.stderr).splitlines() if "slicing warnings:" in line]
            detail = message if isinstance(message, str) else message[0]
            if reasons:
                detail += " (" + "; ".join(reasons) + ")"
            return SliceResult(settings_summary=summary, error=detail)

        plate_info = (result.get("sliced_plates") or [{}])[0]
        warnings = [w for w in [plate_info.get("warning_message")] if w]
        filaments = plate_info.get("filaments") or []
        filament_g = sum(float(f.get("total_used_g") or 0) for f in filaments) or None

        nozzle = float(profiles["machine"]["nozzle_diameter"][0])
        total, uncovered = unprinted_features(mesh, parse_gcode_extrusions(gcode), line_width_mm=nozzle * 1.05)
        if total and uncovered / total > config.SLICE_CHECK_WARN_FRACTION:
            warnings.append(
                f"The slicer prints no plastic for {uncovered} of {total} small features "
                f"({uncovered / total:.0%}) — mostly fine tips that will come out shorter or missing. "
                "Print larger or add wall thickening to keep them."
            )

        shutil.move(tmp / PROJECT_FILENAME, work_dir / PROJECT_FILENAME)
        return SliceResult(
            project_file=PROJECT_FILENAME,
            print_time_s=float(plate_info["total_predication"]) if plate_info.get("total_predication") else None,
            filament_g=filament_g,
            layer_height_mm=float(profiles["process"]["layer_height"]),
            settings_summary=summary,
            warnings=warnings,
            features_total=total,
            features_uncovered=uncovered,
        )

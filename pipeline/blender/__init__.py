"""Headless Blender as an isolated subprocess worker — the same pattern as
mflux (text-to-image) and TRELLIS.2 (image-to-3D): Blender ships its own
Python 3.13 with modules this project's venv doesn't have (`openvdb`,
`bpy`/`bmesh`), so it's called as `blender -b --python <script> -- <args>`
rather than imported.

Data crosses the boundary as .npz files (vertices `v`, faces `f`) — Blender's
bundled numpy reads/writes them directly, and it avoids any STL/PLY
re-processing ambiguity (see the STL-export incident in CLAUDE.md: reloading
with default processing once silently damaged topology).

Scripts live in pipeline/blender/scripts/ and must NOT import `pipeline` —
they run inside Blender's interpreter, which doesn't have this project's
dependencies.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import trimesh

from pipeline import config

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"


class BlenderUnavailableError(RuntimeError):
    """Blender isn't installed where config/MESHY_BLENDER says it is."""


class BlenderScriptError(RuntimeError):
    """A Blender script ran but exited non-zero or timed out."""


def find_blender() -> str:
    candidate = os.environ.get("MESHY_BLENDER") or config.BLENDER_EXECUTABLE
    if Path(candidate).is_file():
        return candidate
    on_path = shutil.which("blender")
    if on_path:
        return on_path
    raise BlenderUnavailableError(
        f"Blender not found at {candidate!r} (set MESHY_BLENDER or config.BLENDER_EXECUTABLE)"
    )


def is_available() -> bool:
    try:
        find_blender()
        return True
    except BlenderUnavailableError:
        return False


def run_script(script_name: str, *args: str | Path, timeout_s: float | None = None) -> str:
    """Runs pipeline/blender/scripts/<script_name> headlessly; returns stdout.

    --factory-startup ignores the user's Blender prefs/add-ons so results
    don't depend on whatever is configured in the GUI; --python-exit-code
    makes an uncaught Python exception a non-zero exit instead of Blender's
    default of logging it and exiting 0.
    """
    blender = find_blender()
    cmd = [
        blender,
        "-b",
        "--factory-startup",
        "-noaudio",
        "--python-exit-code",
        "1",
        "--python",
        str(SCRIPTS_DIR / script_name),
        "--",
        *[str(a) for a in args],
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s or config.BLENDER_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        raise BlenderScriptError(f"{script_name} timed out after {exc.timeout}s") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        raise BlenderScriptError(f"{script_name} failed (exit {proc.returncode}):\n{tail}")
    return proc.stdout


def save_npz(mesh: trimesh.Trimesh, path: Path) -> Path:
    np.savez(path, v=np.asarray(mesh.vertices, dtype=np.float64), f=np.asarray(mesh.faces, dtype=np.int64))
    return path


def load_npz(path: Path) -> trimesh.Trimesh:
    """process=False, then a plain exact-position weld: the Blender scripts
    emit indexed meshes, so this is a safe no-op-ish merge (no tolerance-
    based dedup that could fuse distinct nearby vertices)."""
    with np.load(path) as d:
        m = trimesh.Trimesh(vertices=d["v"], faces=d["f"], process=False)
    m.merge_vertices(digits_vertex=8)
    return trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False)

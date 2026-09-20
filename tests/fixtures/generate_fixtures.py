"""One-time generator for the synthetic broken meshes used by
test_repair_stage.py. Re-run manually (`uv run python3 tests/fixtures/generate_fixtures.py`)
if the fixtures ever need to be regenerated; the .stl files themselves are
committed so tests don't depend on re-running this.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

FIXTURES_DIR = Path(__file__).resolve().parent


def make_hole_mesh() -> trimesh.Trimesh:
    """A box with one face's two triangles removed: an open boundary."""
    box = trimesh.creation.box(extents=(10, 10, 10))
    # Each box face is 2 triangles; drop the first face (faces 0-1) to punch a hole.
    keep = np.ones(len(box.faces), dtype=bool)
    keep[0:2] = False
    box.update_faces(keep)
    box.remove_unreferenced_vertices()
    return box


def make_flipped_normals_mesh() -> trimesh.Trimesh:
    """A box with a subset of face windings reversed (inconsistent normals)."""
    box = trimesh.creation.box(extents=(10, 10, 10))
    faces = box.faces.copy()
    # Reverse winding on half the faces.
    faces[::2] = faces[::2, ::-1]
    return trimesh.Trimesh(vertices=box.vertices, faces=faces, process=False)


def make_multi_shell_mesh() -> trimesh.Trimesh:
    """Two disjoint boxes: a large primary body plus a tiny floater fragment."""
    big = trimesh.creation.box(extents=(10, 10, 10))
    small = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
    small.apply_translation((20, 0, 0))
    return trimesh.util.concatenate([big, small])


def make_thin_wall_mesh() -> trimesh.Trimesh:
    """A hollow box shell whose wall thickness (0.5mm here, at true scale) is
    below the 1.2mm FDM floor once scaled to mm — built as two nested boxes
    with the inner one inverted (a boolean-difference shell)."""
    outer = trimesh.creation.box(extents=(20, 20, 20))
    inner = trimesh.creation.box(extents=(19.0, 19.0, 19.0))  # 0.5mm wall each side
    shell = outer.difference(inner, engine="manifold")
    return shell


if __name__ == "__main__":
    fixtures = {
        "hole_cube.stl": make_hole_mesh(),
        "flipped_normals_cube.stl": make_flipped_normals_mesh(),
        "multi_shell.stl": make_multi_shell_mesh(),
        "thin_wall.stl": make_thin_wall_mesh(),
    }
    for filename, mesh in fixtures.items():
        path = FIXTURES_DIR / filename
        mesh.export(str(path))
        print(f"Wrote {path} ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")

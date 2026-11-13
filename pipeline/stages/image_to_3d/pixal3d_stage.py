"""Image-to-3D stage backed by Pixal3D (TencentARC, MIT) via the
`lyonsno/pixal3d-mlx` port (vendor/pixal3d-mlx/, not committed — see
scripts/setup_pixal3d.sh). Optional; TRELLIS.2 stays the default.

Compared with TRELLIS.2 on three images through the full pipeline
(research/model-refresh-2026-11.md, 2026-11-05): Pixal3D estimates the
camera (MoGe-2) and reconstructs in that camera's frame, so it keeps the
input's pose and proportions noticeably better — e.g. an upright Godzilla
with the curled tail from the concept, where TRELLIS.2 leaned it forward.
But it invents the unseen sides worse (slab tails, a thin dorsal fin),
its surfaces are bumpier, and it takes ~7.5 min per run at 1024 (~2x
TRELLIS.2; 13GB peak). Best for a reference photo whose pose matters,
checked in the preview before printing.

Reuses Trellis2Stage's BiRefNet cut-out (run in the TRELLIS venv, which has
rembg) and best-of-N loop; only the generator differs. One candidate by
default, since each one costs ~7.5 min.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from pipeline import config
from pipeline.stages.image_to_3d.trellis2_stage import Trellis2Stage
from pipeline.types import GeneratedImage

PIXAL3D_DIR = config.VENDOR_DIR / "pixal3d-mlx"

# The port exports in upstream Pixal3D's frame, in which the input camera
# looks from glTF -Z. Turning it 180° about the vertical (glTF Y) axis makes
# the slicer's/preview's front view the input image's view.
FACE_CAMERA = np.diag([-1.0, 1.0, -1.0, 1.0])


def _pixal3d_python() -> Path:
    override = os.environ.get("MESHY_PIXAL3D_PYTHON")
    if override:
        return Path(override)
    # Sibling conda env of the one running this process (…/envs/meshy_local).
    prefix = Path(sys.prefix)
    if prefix.parent.name == "envs":
        return prefix.parent / config.PIXAL3D_CONDA_ENV / "bin" / "python"
    return Path(sys.prefix) / "envs" / config.PIXAL3D_CONDA_ENV / "bin" / "python"


def _turn_to_face_camera(glb_path: Path) -> None:
    import trimesh

    scene = trimesh.load(glb_path, force="scene")
    scene.apply_transform(FACE_CAMERA)
    scene.export(glb_path)


class Pixal3DStage(Trellis2Stage):
    SOURCE_MODEL = "TencentARC/Pixal3D (via pixal3d-mlx)"
    MODEL_NAME = "pixal3d-mlx"

    def __init__(
        self,
        resolution: int = 1024,
        target_faces: int = 200_000,
        texture_size: int = 1024,
        seed: int | None = None,
    ) -> None:
        super().__init__(resolution=resolution, target_faces=target_faces,
                         texture_size=texture_size, seed=seed)
        self.python = _pixal3d_python()
        self.generate_script = PIXAL3D_DIR / "generate_pixal3d.py"
        if not self.python.exists() or not self.generate_script.exists():
            raise RuntimeError(
                f"Pixal3D isn't set up (looked for {self.python} and {self.generate_script}). "
                f"Run scripts/setup_pixal3d.sh, or set MESHY_PIXAL3D_PYTHON."
            )

    def default_num_candidates(self) -> int:
        return config.PIXAL3D_NUM_CANDIDATES

    def _generation_metadata(self) -> dict:
        return {
            "resolution": self.resolution,
            "target_faces": self.target_faces,
            "texture_size": self.texture_size,
            "camera": "MoGe-2 estimate, model turned to face the input camera",
        }

    def _generate_one(self, image: GeneratedImage, out_path: Path, seed: int | None) -> None:
        cmd = [
            str(self.python), str(self.generate_script),
            "--image", str(image.image_path),
            "--output", str(out_path),
            "--resolution", str(self.resolution),
            "--target-faces", str(self.target_faces),
            "--texture-size", str(self.texture_size),
        ]
        if seed is not None:
            cmd += ["--seed", str(seed)]
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=PIXAL3D_DIR,
            env={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
        if result.returncode != 0 or not out_path.exists():
            raise RuntimeError(
                f"pixal3d-mlx generation failed (exit {result.returncode}):\n{result.stderr[-4000:]}"
            )
        if "MoGe MLX camera" not in result.stdout + result.stderr:
            # The port silently falls back to a 49° default FOV without MoGe.
            print("  Warning: Pixal3D ran without MoGe camera estimation (default FOV).")
        _turn_to_face_camera(out_path)

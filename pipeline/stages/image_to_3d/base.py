from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pipeline.types import GeneratedImage, RawMesh


class ImageTo3DStage(Protocol):
    """Reconstructs a 3D mesh from a single image. Swappable implementation:
    sf3d_stage.py today; hunyuan3d_stage.py / trellis2_stage.py later."""

    def reconstruct(
        self, image: GeneratedImage, work_dir: Path, num_candidates: int | None = None
    ) -> RawMesh: ...

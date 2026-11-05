from __future__ import annotations

from pathlib import Path
from typing import Protocol


class PreviewStage(Protocol):
    """Renders images of the final exported STL into work_dir and returns
    their filenames (relative to work_dir). Best-effort: returns [] rather
    than raising if rendering is unavailable — previews must never fail a
    job whose STL is already written."""

    def render(self, stl_path: Path, work_dir: Path) -> list[str]: ...

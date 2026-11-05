from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pipeline.types import SliceResult


class SliceStage(Protocol):
    """Slices the final exported STL for the target printer. Best-effort:
    returns a SliceResult with `error` set rather than raising — slicing must
    never fail a job whose STL is already written."""

    def slice(self, stl_path: Path, work_dir: Path) -> SliceResult: ...

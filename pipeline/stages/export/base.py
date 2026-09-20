from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pipeline.types import PrintReadyMesh, STLResult


class ExportStage(Protocol):
    """Writes a repaired, already-scaled mesh out to STL, and performs a
    final sanity re-check of the exported file's bounding box (catches
    unit-conversion bugs, e.g. glTF meters vs. STL/slicer millimeters,
    that could otherwise slip through silently)."""

    def export(self, mesh: PrintReadyMesh, work_dir: Path) -> STLResult: ...

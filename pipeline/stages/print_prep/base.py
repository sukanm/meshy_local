from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pipeline.types import PrintReadyMesh


class PrintPrepStage(Protocol):
    """Turns a repaired, watertight, scaled mesh into something physically
    printable on the target FDM printer: wall thickening, a print base, and
    placement on the build plate. Runs between repair and export.

    wall_offset_mm: None = auto (thicken only if walls are likely to print as
    gaps), 0 = off, >0 = always offset by exactly this much.
    base_mode: one of config.BASE_MODES, or None for config.BASE_MODE_DEFAULT.
    """

    def prepare(
        self,
        mesh: PrintReadyMesh,
        work_dir: Path,
        wall_offset_mm: float | None = None,
        base_mode: str | None = None,
    ) -> PrintReadyMesh: ...

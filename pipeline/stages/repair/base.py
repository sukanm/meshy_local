from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pipeline.types import PrintReadyMesh, RawMesh


class RepairStage(Protocol):
    """Validates and repairs a raw generated mesh into print-ready geometry.

    First-class pipeline stage (see CLAUDE.md and
    research/meshy-ai-research.md section 5) — not a bolted-on post-process.
    Must be independently invokable on any mesh, not only pipeline output.
    """

    def process(
        self,
        mesh: RawMesh,
        target_longest_dimension_mm: float,
        work_dir: Path,
        smoothing_iterations: int | None = None,
        voxel_resolution: int | None = None,
    ) -> PrintReadyMesh: ...

"""Typed contracts passed between pipeline stages.

Plain dataclasses on purpose: the pipeline package must stay free of any
web-framework (FastAPI/pydantic) or CLI dependency so stages stay swappable
and testable in isolation. `server/schemas.py` mirrors these for the HTTP
layer instead of importing them directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


@dataclass
class TextPrompt:
    prompt: str
    seed: int | None = None
    negative_prompt: str | None = None


@dataclass
class GeneratedImage:
    image_path: Path
    prompt: TextPrompt
    generation_metadata: dict = field(default_factory=dict)


@dataclass
class RawMesh:
    mesh_path: Path
    source_model: str
    source_format: str  # "glb" | "obj" | "stl" | ...
    generation_metadata: dict = field(default_factory=dict)


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    BLOCKING = "blocking"


@dataclass
class Issue:
    code: str
    message: str
    severity: Severity


class PrintReadinessStatus(str, Enum):
    PRINT_READY = "PRINT_READY"
    PRINT_READY_WITH_WARNINGS = "PRINT_READY_WITH_WARNINGS"
    NEEDS_MANUAL_REPAIR = "NEEDS_MANUAL_REPAIR"


@dataclass
class RepairReport:
    is_watertight: bool
    was_repaired: bool
    actions_taken: list[str] = field(default_factory=list)
    remaining_issues: list[Issue] = field(default_factory=list)
    min_wall_thickness_mm: float | None = None
    wall_thickness_below_floor_fraction: float | None = None
    # Fraction of sampled points below one nozzle width (config.
    # WALL_THICKNESS_CRITICAL_MM) — likely an actual gap/hole when sliced,
    # not just a fragile-but-present wall. See trimesh_repair.py's tiered
    # thin-wall message for how this is surfaced alongside the field above.
    wall_thickness_critical_fraction: float | None = None
    bbox_mm: tuple[float, float, float] | None = None
    fits_build_volume: bool | None = None
    status: PrintReadinessStatus = PrintReadinessStatus.NEEDS_MANUAL_REPAIR


@dataclass
class PrintReadyMesh:
    mesh_path: Path
    repair_report: RepairReport


@dataclass
class STLResult:
    stl_path: Path
    repair_report: RepairReport


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


@dataclass
class Job:
    job_id: str
    prompt: TextPrompt
    target_longest_dimension_mm: float
    status: JobStatus = JobStatus.PENDING
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    stage_config: dict = field(default_factory=dict)
    current_stage: str | None = None
    generated_image_path: Path | None = None
    raw_mesh_path: Path | None = None
    stl_path: Path | None = None
    repair_report: RepairReport | None = None
    error: str | None = None

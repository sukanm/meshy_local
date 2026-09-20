"""Pydantic request/response models for the HTTP layer. Mirrors
pipeline/types.py but stays a separate, independent set of types — the
pipeline package must not import anything from here (see CLAUDE.md's
architecture plan, section 1: routes are thin, pipeline stays HTTP-agnostic).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=800)
    scale_mm: float = Field(gt=0, le=256, description="Target longest dimension in mm.")
    seed: int | None = None
    image_model: Literal["trellis2", "sf3d"] = "trellis2"
    smoothing_iterations: int | None = Field(default=None, ge=0, le=25)
    voxel_resolution: int | None = Field(default=None, ge=50, le=600)
    num_candidates: int | None = Field(
        default=None, ge=1, le=8, description="TRELLIS.2 only; ignored for sf3d."
    )


class GenerateResponse(BaseModel):
    job_id: str

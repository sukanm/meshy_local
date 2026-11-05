"""Pydantic request/response models for the HTTP layer. Mirrors
pipeline/types.py but stays a separate, independent set of types — the
pipeline package must not import anything from here (see CLAUDE.md's
architecture plan, section 1: routes are thin, pipeline stays HTTP-agnostic).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class GenerateRequest(BaseModel):
    # Optional when a reference image is given (it's then just the job's label).
    prompt: str = Field(default="", max_length=800)
    scale_mm: float = Field(gt=0, le=256, description="Target longest dimension in mm.")
    seed: int | None = None
    image_model: Literal["trellis2", "sf3d"] = "trellis2"
    smoothing_iterations: int | None = Field(default=None, ge=0, le=25)
    voxel_resolution: int | None = Field(default=None, ge=50, le=600)
    num_candidates: int | None = Field(
        default=None, ge=1, le=8, description="TRELLIS.2 only; ignored for sf3d."
    )
    wall_offset_mm: float | None = Field(
        default=None, ge=0, le=2, description="None = auto, 0 = off, >0 = outward offset in mm."
    )
    base_mode: Literal["flatten", "plinth", "none"] | None = None
    reference_image_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{32}$",
        description="A staged reference image (from /reference-image*) to use instead of generating one.",
    )

    @model_validator(mode="after")
    def _prompt_or_image(self) -> "GenerateRequest":
        if not self.prompt.strip() and not self.reference_image_id:
            raise ValueError("Enter a prompt or choose a reference image.")
        return self


class ReferenceUrlRequest(BaseModel):
    url: str = Field(min_length=8, max_length=4000)
    # A search result's thumbnail, tried if the full image is blocked.
    fallback_url: str | None = Field(default=None, max_length=4000)
    title: str | None = Field(default=None, max_length=500)
    page_url: str | None = Field(default=None, max_length=4000)
    provider: str | None = Field(default=None, max_length=40)


class ReferenceImageResponse(BaseModel):
    ref_id: str
    image_url: str
    width: int
    height: int


class GenerateResponse(BaseModel):
    job_id: str

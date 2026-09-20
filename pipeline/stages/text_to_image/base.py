from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pipeline.types import GeneratedImage, TextPrompt


class TextToImageStage(Protocol):
    """Turns a text prompt into a single concept image for the reconstruction
    stage to condition on. Swappable implementation, e.g. mflux_stage.py."""

    def generate(self, prompt: TextPrompt, work_dir: Path) -> GeneratedImage: ...

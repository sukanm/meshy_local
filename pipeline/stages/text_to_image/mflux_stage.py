"""Text-to-image stage backed by mflux's Z-Image Turbo (MLX-native, Apple
Silicon).

Installed as an isolated `uv tool install mflux` rather than a dependency of
this project: mflux pins transformers>=5.0, which conflicts with Stable
Fast 3D's hard pin of transformers==4.42.3 in the shared project venv (see
research/local-text-to-3d-models-m4-max.md and CLAUDE.md's dependency
section). Calling it as a subprocess is the documented fallback for a
genuine cross-stage dependency conflict, keeping each stage's environment
independent.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pipeline import config
from pipeline.types import GeneratedImage, TextPrompt


class MFluxTextToImageStage:
    def __init__(self, executable: str = config.MFLUX_EXECUTABLE) -> None:
        resolved = shutil.which(executable)
        if resolved is None:
            raise RuntimeError(
                f"'{executable}' not found on PATH. Install it with "
                f"`uv tool install mflux` (see pipeline/config.py)."
            )
        self.executable = resolved

    def generate(self, prompt: TextPrompt, work_dir: Path) -> GeneratedImage:
        work_dir.mkdir(parents=True, exist_ok=True)
        output_path = work_dir / "concept_image.png"
        full_prompt = prompt.prompt + config.IMAGE_GEN_PROMPT_SUFFIX

        cmd = [
            self.executable,
            "--prompt",
            full_prompt,
            "--steps",
            str(config.IMAGE_GEN_STEPS),
            "--width",
            str(config.IMAGE_GEN_SIZE_PX),
            "--height",
            str(config.IMAGE_GEN_SIZE_PX),
            "--quantize",
            str(config.IMAGE_GEN_QUANTIZE_BITS),
            "--no-metadata",
            "--output",
            str(output_path),
        ]
        if prompt.negative_prompt:
            cmd += ["--negative-prompt", prompt.negative_prompt]
        if prompt.seed is not None:
            cmd += ["--seed", str(prompt.seed)]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0 or not output_path.exists():
            raise RuntimeError(
                f"mflux text-to-image generation failed (exit "
                f"{result.returncode}):\n{result.stderr[-4000:]}"
            )

        return GeneratedImage(
            image_path=output_path,
            prompt=prompt,
            generation_metadata={
                "model": "Tongyi-MAI/Z-Image-Turbo",
                "steps": config.IMAGE_GEN_STEPS,
                "size_px": config.IMAGE_GEN_SIZE_PX,
                "quantize_bits": config.IMAGE_GEN_QUANTIZE_BITS,
                "full_prompt": full_prompt,
            },
        )

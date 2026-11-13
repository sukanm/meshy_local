"""Concept images for review before 3D (see config.CONCEPT_BATCH_SIZE).

One concept is made per call so the web UI can show each as it finishes and
stop as soon as one is good. The image is staged exactly like a reference
image (pipeline.reference_images), so building from it later is the existing
reference-image path; its source records the prompt and seed it came from.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

from pipeline import config, reference_images
from pipeline.stages.text_to_image.base import TextToImageStage
from pipeline.types import TextPrompt


def make_concept(
    text_to_image: TextToImageStage,
    prompt: str,
    seed: int | None = None,
    directory: Path | None = None,
) -> tuple[str, int]:
    """Generates one concept image and stages it. Returns (ref_id, seed).

    A seed is always chosen here (random if not given) so the pick can be
    reproduced and the UI can offer the next seeds after it."""
    directory = directory or config.REFERENCE_IMAGE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    with tempfile.TemporaryDirectory(dir=directory, prefix="_concept_") as tmp:
        image = text_to_image.generate(TextPrompt(prompt=prompt, seed=seed), Path(tmp))
        meta = image.generation_metadata or {}
        source = {
            "kind": "concept",
            "title": prompt[:500],
            "prompt": prompt,
            "seed": seed,
            "model": meta.get("model"),
            "full_prompt": meta.get("full_prompt"),
        }
        ref_id = reference_images.stage_image_bytes(
            image.image_path.read_bytes(), {k: v for k, v in source.items() if v is not None}, directory
        )
    return ref_id, seed

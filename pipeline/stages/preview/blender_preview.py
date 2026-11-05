"""Preview stage: headless Blender renders of the final exported STL.

Renders the actual file the slicer will receive (not an intermediate mesh),
so defects that only exist after the last processing step are visible. This
is exactly the gap that let job eeaae724ce2b's long "starburst" sliver rods
reach a physical print: earlier spot-check renders were of pre-snap
intermediate meshes, and the rods only existed in the final STL.

Best-effort by design (see PreviewStage): any failure returns [] and the job
still succeeds — the STL is already written by the time this runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pipeline import blender, config

log = logging.getLogger(__name__)


class BlenderPreviewStage:
    def render(self, stl_path: Path, work_dir: Path) -> list[str]:
        views = [f"{name}:{az}:{el}" for name, az, el in config.PREVIEW_VIEWS]
        try:
            blender.run_script(
                "render_previews.py", stl_path, work_dir, config.PREVIEW_RESOLUTION_PX, *views
            )
        except (blender.BlenderUnavailableError, blender.BlenderScriptError) as exc:
            log.warning("Preview rendering skipped: %s", exc)
            return []
        names = [f"preview_{name}.png" for name, _az, _el in config.PREVIEW_VIEWS]
        return [n for n in names if (work_dir / n).exists()]

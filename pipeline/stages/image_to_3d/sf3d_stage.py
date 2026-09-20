"""Image-to-3D stage backed by Stability AI's Stable Fast 3D, vendored at
vendor/stable-fast-3d/ (official first-party MPS support — see
research/local-text-to-3d-models-m4-max.md).

Import order and env vars here mirror scripts/verify_sf3d_standalone.py,
which is the standing Phase 0 health check for this same code path — keep
the two in sync if either changes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from pipeline import config  # noqa: E402
from pipeline.types import GeneratedImage, RawMesh  # noqa: E402

SF3D_DIR = config.VENDOR_DIR / "stable-fast-3d"

# Must import these before SF3D_DIR touches sys.path: SF3D_DIR contains
# sibling `uv_unwrapper/` and `texture_baker/` *project* directories whose
# bare names collide with the real installed packages of the same name as
# namespace-package portions. Importing first populates sys.modules, which
# Python's import statement always checks before any path finder, so SF3D's
# own `from uv_unwrapper import Unwrapper` reuses these correct modules.
import texture_baker  # noqa: F401,E402
import uv_unwrapper  # noqa: F401,E402

sys.path.append(str(SF3D_DIR))

import rembg  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from sf3d.system import SF3D  # noqa: E402
from sf3d.utils import get_device, remove_background, resize_foreground  # noqa: E402


class StableFast3DStage:
    def __init__(
        self,
        pretrained_model: str = "stabilityai/stable-fast-3d",
        texture_resolution: int = 1024,
        remesh_option: str = "none",
        target_vertex_count: int = -1,
        foreground_ratio: float = 0.85,
    ) -> None:
        self.device = get_device()
        self.texture_resolution = texture_resolution
        self.remesh_option = remesh_option
        self.target_vertex_count = target_vertex_count
        self.foreground_ratio = foreground_ratio

        self.model = SF3D.from_pretrained(
            pretrained_model,
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
        self.model.to(self.device)
        self.model.eval()
        self._rembg_session = rembg.new_session()

    def reconstruct(
        self, image: GeneratedImage, work_dir: Path, num_candidates: int | None = None
    ) -> RawMesh:
        # num_candidates is a TRELLIS2Stage-specific feature (see
        # trellis2_stage.py): SF3D is feed-forward/deterministic per input,
        # not diffusion-sampled, so there's no seed-diversity axis to
        # exploit here — accepted for protocol compatibility, ignored.
        work_dir.mkdir(parents=True, exist_ok=True)

        pil_image = remove_background(
            Image.open(image.image_path).convert("RGBA"), self._rembg_session
        )
        pil_image = resize_foreground(pil_image, self.foreground_ratio)

        with torch.no_grad():
            mesh, _glob_dict = self.model.run_image(
                [pil_image],
                bake_resolution=self.texture_resolution,
                remesh=self.remesh_option,
                vertex_count=self.target_vertex_count,
            )

        out_path = work_dir / "raw_mesh.glb"
        mesh.export(str(out_path), include_normals=True)

        return RawMesh(
            mesh_path=out_path,
            source_model="stabilityai/stable-fast-3d",
            source_format="glb",
            generation_metadata={
                "device": self.device,
                "texture_resolution": self.texture_resolution,
                "remesh_option": self.remesh_option,
                "target_vertex_count": self.target_vertex_count,
            },
        )

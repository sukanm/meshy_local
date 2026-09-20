"""Phase 0 health-check: confirms Stable Fast 3D runs standalone on this Mac
via its own unmodified code path (sf3d.system.SF3D), independent of our own
pipeline wrapper. Kept permanently as a quick "is the ML environment still
working" smoke test.

Must set KMP_DUPLICATE_LIB_OK before importing torch/numpy/scipy: this Mac's
combination of Homebrew libomp (linked into texture_baker/uv_unwrapper at
build time) and the OpenMP runtime bundled inside the scipy/numpy wheels
causes an "OMP: Error #15" duplicate-runtime abort otherwise. This is the
standard, widely-used workaround for that specific conflict, not a general
fix for OpenMP issues.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SF3D_DIR = PROJECT_ROOT / "vendor" / "stable-fast-3d"

# Import the real editable-installed packages *before* SF3D_DIR ever touches
# sys.path. SF3D_DIR contains sibling `uv_unwrapper/` and `texture_baker/`
# *project* directories (setup.py, README, etc.) whose bare names collide
# with the real installed packages of the same name; Python's PathFinder
# treats those bare directories as namespace-package portions and resolves
# to them (empty, no `Unwrapper`/`_C` symbols) before ever consulting the
# editable-install's own meta_path finder, which is registered *after*
# PathFinder. Importing here first populates sys.modules, and Python's
# import statement always checks sys.modules before any finder, so later
# `from uv_unwrapper import ...` calls inside sf3d's own code reuse these
# correct, already-imported modules instead of re-resolving the name.
import texture_baker  # noqa: F401,E402
import uv_unwrapper  # noqa: F401,E402

sys.path.append(str(SF3D_DIR))

import rembg  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from sf3d.system import SF3D  # noqa: E402
from sf3d.utils import get_device, remove_background, resize_foreground  # noqa: E402


def main() -> None:
    device = get_device()
    print(f"Device: {device}")
    print(f"torch: {torch.__version__}")

    example_image = SF3D_DIR / "demo_files" / "examples" / "chair1.png"
    if not example_image.exists():
        raise SystemExit(f"Example image not found: {example_image}")

    verify_output_dir = PROJECT_ROOT / "output" / "_verify_sf3d"
    verify_output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading SF3D model from Hugging Face (stabilityai/stable-fast-3d)...")
    t0 = time.time()
    model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.to(device)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")

    rembg_session = rembg.new_session()
    image = remove_background(Image.open(example_image).convert("RGBA"), rembg_session)
    image = resize_foreground(image, 0.85)
    image.save(verify_output_dir / "input.png")

    print("Running reconstruction...")
    t0 = time.time()
    with torch.no_grad():
        mesh, _glob_dict = model.run_image(
            [image],
            bake_resolution=1024,
            remesh="none",
            vertex_count=-1,
        )
    elapsed = time.time() - t0
    print(f"Reconstruction finished in {elapsed:.1f}s")

    out_path = verify_output_dir / "mesh.glb"
    mesh.export(str(out_path), include_normals=True)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print("SF3D standalone verification: PASSED")


if __name__ == "__main__":
    main()

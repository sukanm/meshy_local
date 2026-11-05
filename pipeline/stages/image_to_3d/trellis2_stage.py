"""Image-to-3D stage backed by TRELLIS.2 via the vendored `trellis2mlx` port
(vendor/trellis2mlx/, MLX-native, MIT-licensed porting code — see
research/local-text-to-3d-models-m4-max.md section 5).

Adopted as the default over Stable Fast 3D after a direct comparison: SF3D
encodes features like eyes purely as flat texture (invisible in the
texture-less STL output), while TRELLIS.2 carves them as real geometry.
TRELLIS.2 is also markedly faster (~2-9 min vs SF3D's ~15 min here).

Runs in its own `uv`-managed venv (vendor/trellis2mlx/.venv, Python 3.11)
and is called via subprocess, the same pattern as MFluxTextToImageStage —
trellis2mlx pins its own dependency set independent of the main project
venv, and subprocess isolation avoids any risk of conflicting with SF3D's
exact pins (transformers==4.42.3, torch nightly) if both stages are ever
imported in the same process.

Note: real TRELLIS.2 output has needed the repair stage's full escalation
ladder (including the voxel-remesh tier) in testing — its raw mesh cleanup
can leave non-manifold geometry that even PyMeshLab's own filters can't
handle (see pipeline/stages/repair/trimesh_repair.py's exception handling
around _pymeshlab_repair, added specifically because of this).

Best-of-N candidate selection (2026-09-18): TRELLIS.2's reconstruction is
image-conditioned, not free-form — different seeds on the same input image
produce visually near-identical shapes (same silhouette/pose/spike layout,
confirmed by direct rendered comparison), but measurably different
*reconstruction cleanliness*. Verified directly on a real test image: two
seeds gave boundary-edge counts of 13,623 vs 6,855 and wall-thickness
below-floor fractions of 87.0% vs 22.2% for the *same* source image — a far
bigger lever than any single generation parameter tuned this session (e.g.
doubling --steps only moved non-manifold edges by ~20% at ~2x the time
cost). Since generation is fully local and has no per-run credit cost
(unlike Meshy, which likely benefits from users regenerating a few times
before committing to the expensive texture pass in their own preview/refine
workflow), generating several candidates and automatically keeping the
cleanest is a safe, mechanical way to capture that same benefit without
requiring the user to visually judge each one.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path

from pipeline import config
from pipeline.types import GeneratedImage, RawMesh

TRELLIS2_DIR = config.VENDOR_DIR / "trellis2mlx"
REMOVE_BACKGROUND_SCRIPT = Path(__file__).with_name("remove_background.py")
CUTOUT_FILENAME = "input_cutout.png"


def _has_transparency(path: Path) -> bool:
    """Whether an image already carries a cut-out (any alpha below 255) —
    the same test generate.py uses to decide whether to skip its own
    background removal."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as img:
        if img.mode != "RGBA":
            return False
        return bool((np.asarray(img)[:, :, 3] < 255).any())


def _topology_defect_count(mesh_path: Path) -> int:
    """Fast proxy for "how messy is this raw reconstruction" — boundary +
    non-manifold edge count, without running the full repair pipeline
    (which would defeat the point of scoring candidates cheaply). Lower is
    better/cleaner. Uses the same edge-multiplicity technique established
    throughout this project's repair/export code."""
    import numpy as np
    import trimesh

    raw = trimesh.load(mesh_path, force="mesh")
    m = trimesh.Trimesh(vertices=raw.vertices, faces=raw.faces, process=False)
    m.process(validate=True)
    edges_sorted = np.sort(m.edges, axis=1)
    _unique, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary = int((counts == 1).sum())
    nonmanifold = int((counts > 2).sum())
    return boundary + nonmanifold


class Trellis2Stage:
    def __init__(
        self,
        resolution: int = 1024,
        target_faces: int = 200_000,
        steps: int = 12,
        texture_size: int = 1024,
        seed: int | None = None,
    ) -> None:
        self.python = TRELLIS2_DIR / ".venv" / "bin" / "python3"
        self.generate_script = TRELLIS2_DIR / "generate.py"
        if not self.python.exists():
            raise RuntimeError(
                f"trellis2mlx venv not found at {self.python}. Set it up per "
                f"vendor/trellis2mlx/README.md (uv venv .venv --python python3.11 "
                f"&& uv pip install -e .)."
            )
        self.resolution = resolution
        self.target_faces = target_faces
        self.steps = steps
        self.texture_size = texture_size
        self.seed = seed

    def _generate_one(self, image: GeneratedImage, out_path: Path, seed: int | None) -> None:
        cmd = [
            str(self.python),
            str(self.generate_script),
            "--image",
            str(image.image_path),
            "--output",
            str(out_path),
            "--resolution",
            str(self.resolution),
            "--target-faces",
            str(self.target_faces),
            "--steps",
            str(self.steps),
            "--texture-size",
            str(self.texture_size),
        ]
        if seed is not None:
            cmd += ["--seed", str(seed)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=TRELLIS2_DIR,
            env={**os.environ, "PYTHONPATH": str(TRELLIS2_DIR)},
        )
        if result.returncode != 0 or not out_path.exists():
            raise RuntimeError(
                f"trellis2mlx generation failed (exit {result.returncode}):\n"
                f"{result.stderr[-4000:]}"
            )

    def _cut_out_background(self, image: GeneratedImage, work_dir: Path) -> tuple[GeneratedImage, dict]:
        """Cuts the object out once per job with config.BACKGROUND_REMOVAL_MODEL
        and returns the RGBA cut-out to reconstruct from (generate.py then
        skips its own u2net pass, for every candidate). Never fails the job:
        on any problem it returns the original image and generate.py removes
        the background itself, as before."""
        model = config.BACKGROUND_REMOVAL_MODEL
        if not model:
            return image, {"model": "generate.py default (u2net)"}
        if _has_transparency(image.image_path):
            return image, {"model": "none — input already has transparency"}
        cutout_path = work_dir / CUTOUT_FILENAME
        try:
            result = subprocess.run(
                [str(self.python), str(REMOVE_BACKGROUND_SCRIPT), str(image.image_path), str(cutout_path),
                 "--model", model],
                capture_output=True,
                text=True,
                timeout=600,  # first use downloads the model (~1GB)
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"  Background removal with {model} failed ({exc}); generate.py will use u2net.")
            return image, {"model": "generate.py default (u2net)", "error": f"{model}: {exc}"}
        if result.returncode != 0 or not cutout_path.exists():
            detail = (result.stderr.strip().splitlines() or ["no output"])[-1]
            print(f"  Background removal with {model} failed ({detail}); generate.py will use u2net.")
            return image, {"model": "generate.py default (u2net)", "error": f"{model}: {detail}"}
        info = json.loads(result.stdout.strip().splitlines()[-1])
        print(f"  Background removed with {model} in {info['load_s'] + info['run_s']:.1f}s")
        return GeneratedImage(image_path=cutout_path, prompt=image.prompt,
                              generation_metadata=image.generation_metadata), info

    def reconstruct(
        self, image: GeneratedImage, work_dir: Path, num_candidates: int | None = None
    ) -> RawMesh:
        work_dir.mkdir(parents=True, exist_ok=True)
        out_path = work_dir / "raw_mesh.glb"
        image, background_removal = self._cut_out_background(image, work_dir)

        if num_candidates is None:
            num_candidates = config.TRELLIS2_NUM_CANDIDATES

        if num_candidates <= 1:
            self._generate_one(image, out_path, self.seed)
            return RawMesh(
                mesh_path=out_path,
                source_model="microsoft/TRELLIS.2-4B (via trellis2mlx)",
                source_format="glb",
                generation_metadata={
                    "resolution": self.resolution,
                    "target_faces": self.target_faces,
                    "steps": self.steps,
                    "texture_size": self.texture_size,
                    "background_removal": background_removal,
                },
            )

        # Distinct, reproducible-if-seeded candidate seeds: reusing the same
        # seed for every candidate would defeat the point (identical output
        # each time), and generate.py itself defaults to a fixed seed (42)
        # when none is given, so a fresh random base is picked explicitly
        # here rather than relying on subprocess-level randomness.
        base_seed = self.seed if self.seed is not None else random.randint(0, 2**31 - 1)
        candidate_seeds = [base_seed + i for i in range(num_candidates)]

        scored: list[tuple[int, int, Path]] = []  # (defect_count, seed, path)
        errors: list[str] = []
        for i, seed in enumerate(candidate_seeds):
            candidate_path = work_dir / f"_candidate_{i}_seed{seed}.glb"
            print(f"  [candidate {i + 1}/{num_candidates}] generating with seed={seed}...")
            try:
                self._generate_one(image, candidate_path, seed)
            except RuntimeError as exc:
                errors.append(f"seed {seed}: {exc}")
                continue
            defects = _topology_defect_count(candidate_path)
            print(f"  [candidate {i + 1}/{num_candidates}] seed={seed} -> {defects} topology defects")
            scored.append((defects, seed, candidate_path))

        if not scored:
            raise RuntimeError(
                f"All {num_candidates} trellis2mlx candidate generations failed:\n" + "\n".join(errors)
            )

        scored.sort(key=lambda t: t[0])
        best_defects, best_seed, best_path = scored[0]
        best_path.replace(out_path)
        for _defects, _seed, path in scored[1:]:
            path.unlink(missing_ok=True)

        selection_report = {
            "num_candidates_requested": num_candidates,
            "num_candidates_succeeded": len(scored),
            "candidate_seeds": candidate_seeds,
            "candidate_defect_counts": {str(seed): defects for defects, seed, _path in scored},
            "selected_seed": best_seed,
            "selected_defect_count": best_defects,
        }
        (work_dir / "candidate_selection.json").write_text(json.dumps(selection_report, indent=2))
        print(
            f"  Selected seed={best_seed} ({best_defects} defects) out of "
            f"{len(scored)}/{num_candidates} successful candidates."
        )

        return RawMesh(
            mesh_path=out_path,
            source_model="microsoft/TRELLIS.2-4B (via trellis2mlx)",
            source_format="glb",
            generation_metadata={
                "resolution": self.resolution,
                "target_faces": self.target_faces,
                "steps": self.steps,
                "texture_size": self.texture_size,
                "background_removal": background_removal,
                "num_candidates": num_candidates,
                "selected_seed": best_seed,
                "candidate_selection": selection_report,
            },
        )

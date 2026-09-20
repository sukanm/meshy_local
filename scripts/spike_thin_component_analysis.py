"""Standalone feasibility spike (not wired into the pipeline): does the
"thin" region of a real voxelized mesh decompose into many small, separate
connected components (isolated spikes/claws/fingers - the case where
selective thickening could work) or one single connected mass covering most
of the surface (the case that already failed once - see
MIN_FEATURE_THICKENING_ENABLED's incident writeup in CLAUDE.md)?

Reuses the exact same thin_mask logic as
pipeline/stages/repair/trimesh_repair.py's _thicken_thin_voxels, then adds
scipy.ndimage.label on top to answer the question above, before spending any
effort building the connected-component-restricted thickening feature for
real.

Usage: uv run python -m scripts.spike_thin_component_analysis <job_id>
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import trimesh
from scipy import ndimage

from pipeline import config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", type=str)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--scale", type=float, default=None, help="Target longest dimension mm (default: source job's own).")
    args = parser.parse_args()

    job_dir = config.OUTPUT_DIR / args.job_id
    job = json.loads((job_dir / "job.json").read_text())
    raw_mesh_path = job["raw_mesh_path"]
    target_mm = args.scale or job["target_longest_dimension_mm"]

    loaded = trimesh.load(raw_mesh_path, force="mesh")
    m = trimesh.Trimesh(vertices=loaded.vertices, faces=loaded.faces, process=False)
    m.merge_vertices()
    m.remove_duplicate_faces()
    m.fix_normals()

    native_extent = (m.bounds[1] - m.bounds[0]).max()
    scale_to_mm = target_mm / native_extent

    resolution = args.resolution or config.adaptive_voxel_resolution(target_mm)
    extent = m.bounds[1] - m.bounds[0]
    pitch = extent.max() / resolution
    pitch_mm = pitch * scale_to_mm
    print(f"job={args.job_id} target_mm={target_mm} resolution={resolution} pitch_mm={pitch_mm:.4f}")

    vox = m.voxelized(pitch=pitch).fill()
    matrix = vox.matrix
    print(f"voxel grid shape={matrix.shape} solid_voxels={int(matrix.sum())}")

    target_radius_voxels = max(1, round((config.FDM_MIN_WALL_THICKNESS_MM / 2.0) / pitch_mm))
    dist_in = ndimage.distance_transform_edt(matrix)
    window = 2 * target_radius_voxels + 1
    local_max = ndimage.maximum_filter(dist_in, size=window)
    thin_mask = matrix & (local_max < target_radius_voxels)

    thin_count = int(thin_mask.sum())
    solid_count = int(matrix.sum())
    print(f"thin voxels: {thin_count} / {solid_count} solid ({thin_count / solid_count:.1%})")

    labeled, num_components = ndimage.label(thin_mask, structure=np.ones((3, 3, 3)))
    sizes = ndimage.sum(thin_mask, labeled, index=np.arange(1, num_components + 1))
    sizes = np.sort(sizes)[::-1]

    print(f"connected components of thin region: {num_components}")
    print(f"largest component: {int(sizes[0])} voxels ({sizes[0] / thin_count:.1%} of all thin voxels)")
    top10 = sizes[:10]
    print(f"top 10 component sizes: {[int(s) for s in top10]}")
    print(f"top 10 as fraction of thin voxels: {top10.sum() / thin_count:.1%}")

    # If we only thickened components below various size caps, how much of
    # the thin region would actually get treated (vs. left as-is because it's
    # part of a large connected mass, e.g. most of the torso surface)?
    for cap_fraction in (0.001, 0.005, 0.01, 0.02, 0.05):
        cap = cap_fraction * solid_count
        treated = sizes[sizes <= cap].sum()
        print(f"  cap={cap_fraction:.1%} of solid voxels ({int(cap)} vox): "
              f"would treat {int(treated)}/{thin_count} thin voxels ({treated / thin_count:.1%}), "
              f"leaving {1 - treated / thin_count:.1%} of the thin region untouched")


if __name__ == "__main__":
    main()

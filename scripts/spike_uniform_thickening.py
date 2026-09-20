"""Standalone feasibility spike (not wired into the pipeline): what happens
if the ENTIRE solid voxel grid is dilated uniformly by a fixed depth, instead
of the selective (and already-abandoned, see MIN_FEATURE_THICKENING_ENABLED)
"only dilate voxels that read as thin" approach?

Rationale: the connected-component spike (scripts/spike_thin_component_analysis.py)
showed the thin region is one single connected mass (99.7% of thin voxels),
so there's no way to selectively target "just the spikes." The remaining
question is whether uniformly growing the *whole* mesh by a fixed radius
predictably reaches the 1.2mm floor everywhere (like the alpha-wrap offset
parameter did, in earlier research) without the volume blowup the selective
approach produced, at the cost of a uniformly chunkier model.

No detail-recovery snap is attempted here: with the whole mesh treated as
"thickened", the existing snap guard (_vertices_near_mask) would block
snapping everywhere anyway, so it would be silently equivalent to skipping
it — this spike skips it explicitly instead, for clarity.

Usage: uv run python -m scripts.spike_uniform_thickening <job_id> [--depths 1 2 3 4]
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import trimesh
from scipy import ndimage

from pipeline import config
from pipeline.stages.export.mesh_topology import remove_slivers, weld_and_repair
from pipeline.stages.repair.thickness import estimate_min_wall_thickness_mm
from pipeline.stages.repair.trimesh_repair import _curvature_adaptive_smooth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", type=str)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()

    job_dir = config.OUTPUT_DIR / args.job_id
    job = json.loads((job_dir / "job.json").read_text())
    raw_mesh_path = job["raw_mesh_path"]
    target_mm = args.scale or job["target_longest_dimension_mm"]

    loaded = trimesh.load(raw_mesh_path, force="mesh")
    m = trimesh.Trimesh(vertices=loaded.vertices, faces=loaded.faces, process=False)
    m.merge_vertices()
    m.update_faces(m.unique_faces())
    m.fix_normals()

    native_extent = (m.bounds[1] - m.bounds[0]).max()
    scale_to_mm = target_mm / native_extent

    resolution = args.resolution or config.adaptive_voxel_resolution(target_mm)
    smoothing_iterations = config.adaptive_smoothing_iterations(resolution)
    extent = m.bounds[1] - m.bounds[0]
    pitch = extent.max() / resolution
    pitch_mm = pitch * scale_to_mm
    print(f"job={args.job_id} target_mm={target_mm} resolution={resolution} pitch_mm={pitch_mm:.4f}\n")

    vox = m.voxelized(pitch=pitch).fill()
    base_matrix = vox.matrix
    base_solid = int(base_matrix.sum())
    print(f"baseline solid voxels: {base_solid}\n")

    for depth in args.depths:
        print(f"--- uniform dilation depth={depth} voxels (~{depth * pitch_mm:.3f}mm added radius) ---")
        # scipy quirk: iterations=0 does NOT mean "no-op" - it means "iterate
        # to convergence", which fills the entire grid. Guard explicitly.
        dilated = base_matrix if depth == 0 else ndimage.binary_dilation(base_matrix, iterations=depth)
        vox_d = trimesh.voxel.VoxelGrid(dilated, transform=vox.transform)
        remeshed = vox_d.marching_cubes
        remeshed.apply_transform(vox_d.transform)

        if len(remeshed.faces) > config.VOXEL_REMESH_MAX_FACES_BEFORE_SIMPLIFY:
            import pymeshlab

            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(vertex_matrix=remeshed.vertices, face_matrix=remeshed.faces.astype(np.int32)))
            ms.meshing_decimation_quadric_edge_collapse(
                targetfacenum=config.VOXEL_REMESH_SIMPLIFY_TARGET_FACES,
                preservetopology=True,
                preserveboundary=True,
            )
            d = ms.current_mesh()
            remeshed = trimesh.Trimesh(vertices=d.vertex_matrix(), faces=d.face_matrix(), process=False)
            remeshed.process(validate=True)
            remeshed.fill_holes()
            remeshed.fix_normals()
            remeshed.merge_vertices()
            remeshed.update_faces(remeshed.nondegenerate_faces())
            remeshed.remove_unreferenced_vertices()

        remeshed = remove_slivers(remeshed, config.SLIVER_EDGE_LENGTH_MULTIPLE)
        if smoothing_iterations > 0:
            remeshed = _curvature_adaptive_smooth(remeshed, smoothing_iterations)

        repaired, is_clean, stats = weld_and_repair(remeshed)
        if is_clean:
            remeshed = repaired

        min_thick, below_floor, below_critical = estimate_min_wall_thickness_mm(
            remeshed,
            scale_to_mm=scale_to_mm,
            sample_count=3000,
            floor_mm=config.FDM_MIN_WALL_THICKNESS_MM,
            critical_mm=config.WALL_THICKNESS_CRITICAL_MM,
        )

        bbox_mm = (remeshed.bounds[1] - remeshed.bounds[0]) * scale_to_mm
        volume_mm3 = abs(remeshed.volume) * (scale_to_mm ** 3) if remeshed.is_watertight else None

        print(f"  watertight={remeshed.is_watertight}  clean_weld={is_clean}")
        print(f"  bbox_mm={tuple(round(x, 2) for x in bbox_mm)}")
        print(f"  volume_mm3={volume_mm3:.1f}" if volume_mm3 is not None else "  volume_mm3=n/a (not watertight)")
        if min_thick is not None:
            print(f"  below 1.2mm floor: {below_floor:.1%}   below nozzle width (0.4mm): {below_critical:.1%}   min={min_thick:.4f}mm")
        else:
            print("  thickness: n/a")
        print()


if __name__ == "__main__":
    main()

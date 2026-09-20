"""Standalone feasibility spike (not wired into the pipeline): re-test CGAL
alpha wrap (pymeshlab.generate_alpha_wrap) as a TRUE surface offset, applied
to the already-clean, already-detailed, watertight mesh a job actually
produced (print_ready_mesh.ply) -- NOT the raw, self-intersecting
reconstruction it was tested against and rejected on earlier (see CLAUDE.md's
"CGAL Alpha Wrap" section: volume was unstable there because alpha wrap was
threading through internal cracks in messy, self-intersecting raw geometry).

The hypothesis this spike checks: with clean, already-manifold input, alpha
wrap's `offset` parameter should behave as a genuine, detail-preserving
Minkowski-style dilation (no cracks to thread through), unlike the crude
voxelize-dilate-remesh approach in spike_uniform_thickening.py, which
necessarily destroys sub-pitch surface detail by construction.

Usage: uv run python -m scripts.spike_alpha_wrap_offset_detailed <job_id> \
    [--alphas A1 A2 ...] [--offsets O1 O2 ...]
(alpha/offset in mm, since the source mesh is already scaled to mm)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pymeshlab
import trimesh

from pipeline import config
from pipeline.stages.export.mesh_topology import weld_and_repair
from pipeline.stages.repair.thickness import estimate_min_wall_thickness_mm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", type=str)
    parser.add_argument("--alphas", type=float, nargs="+", default=[2.0, 1.0])
    parser.add_argument("--offsets", type=float, nargs="+", default=[0.6])
    args = parser.parse_args()

    job_dir = config.OUTPUT_DIR / args.job_id
    source_path = job_dir / "print_ready_mesh.ply"
    reference = trimesh.load(source_path, force="mesh", process=False)
    print(f"source: {source_path}")
    print(f"reference: watertight={reference.is_watertight} faces={len(reference.faces)} "
          f"volume_mm3={abs(reference.volume):.1f} bbox_mm={tuple(round(x,2) for x in (reference.bounds[1]-reference.bounds[0]))}\n")

    for alpha in args.alphas:
        for offset in args.offsets:
            print(f"--- alpha={alpha}mm offset={offset}mm ---")
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(str(source_path))
            try:
                ms.generate_alpha_wrap(
                    alpha=pymeshlab.PureValue(alpha),
                    offset=pymeshlab.PureValue(offset),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED: {exc}\n")
                continue
            wrapped = ms.current_mesh()
            m = trimesh.Trimesh(vertices=wrapped.vertex_matrix(), faces=wrapped.face_matrix(), process=False)

            repaired, is_clean, weld_stats = weld_and_repair(m)
            if is_clean:
                m = repaired

            volume_mm3 = abs(m.volume) if m.is_watertight else None
            bbox_mm = tuple(float(x) for x in (m.bounds[1] - m.bounds[0]))

            min_thick = below_floor = below_critical = None
            if m.is_watertight:
                min_thick, below_floor, below_critical = estimate_min_wall_thickness_mm(
                    m, scale_to_mm=1.0, sample_count=config.WALL_THICKNESS_SAMPLE_COUNT,
                    floor_mm=config.FDM_MIN_WALL_THICKNESS_MM, critical_mm=config.WALL_THICKNESS_CRITICAL_MM,
                )

            # Detail-preservation proxy: mean/max distance from the wrapped
            # surface back to the reference (detailed, pre-wrap) surface. If
            # this tracks the offset value closely and doesn't blow up, the
            # wrap is a genuine uniform surface offset, not something fusing
            # separate features together (see CLAUDE.md's alpha-wrap incident
            # for why that distinction matters).
            closest, dist, _ = reference.nearest.on_surface(m.vertices)
            mean_dist = float(dist.mean())
            max_dist = float(dist.max())

            print(f"  watertight={m.is_watertight} clean_weld={is_clean} faces={len(m.faces)}")
            print(f"  bbox_mm={tuple(round(x,2) for x in bbox_mm)}")
            if volume_mm3 is not None:
                print(f"  volume_mm3={volume_mm3:.1f} (reference={abs(reference.volume):.1f}, "
                      f"ratio={volume_mm3/abs(reference.volume):.2f}x)")
            else:
                print("  volume_mm3=n/a (not watertight)")
            print(f"  mean_dist_to_reference_mm={mean_dist:.4f} max_dist_to_reference_mm={max_dist:.4f}")
            if min_thick is not None:
                print(f"  below 1.2mm floor: {below_floor:.1%}  below nozzle width: {below_critical:.1%}  min={min_thick:.4f}mm")
            else:
                print("  thickness: n/a")
            print()


if __name__ == "__main__":
    main()

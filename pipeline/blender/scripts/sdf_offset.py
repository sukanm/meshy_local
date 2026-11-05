"""Runs INSIDE Blender's bundled Python (has `openvdb`; must not import
`pipeline`). Uniform outward surface offset via an OpenVDB level set.

Usage: blender -b --python sdf_offset.py -- <in.npz> <out.npz> <voxel_mm> <offset_mm> <adaptivity>

Input must be a closed (watertight) mesh: OpenVDB decides inside/outside by
flood fill, which leaks through holes — verified on raw, non-watertight
TRELLIS.2 output, which shattered into thousands of shells. The caller only
invokes this on repair-stage output that's already watertight.
"""

import math
import sys
import time

import numpy as np
import openvdb as vdb

in_path, out_path, voxel, offset, adaptivity = sys.argv[sys.argv.index("--") + 1 :]
voxel, offset, adaptivity = float(voxel), float(offset), float(adaptivity)
t0 = time.time()

with np.load(in_path) as d:
    points = d["v"].astype(np.float32)
    triangles = d["f"].astype(np.int32)

# The narrow band must extend past the offset distance, or the isosurface at
# `offset` would fall outside the stored band and come out empty/clipped.
half_width = max(3.0, math.ceil(offset / voxel) + 3)
grid = vdb.FloatGrid.createLevelSetFromPolygons(
    points,
    triangles=triangles,
    transform=vdb.createLinearTransform(voxelSize=voxel),
    halfWidth=half_width,
)
pts, tris, quads = grid.convertToPolygons(isovalue=offset, adaptivity=adaptivity)
tris = np.asarray(tris, dtype=np.int64).reshape(-1, 3)
quads = np.asarray(quads, dtype=np.int64).reshape(-1, 4)
faces = np.concatenate([tris, quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
# OpenVDB's polygon winding is opposite to trimesh's outward-CCW convention
# (verified: un-flipped output has negative signed volume, which also makes
# the inward-ray wall-thickness estimator cast rays outward).
faces = faces[:, ::-1].copy()

np.savez(out_path, v=np.asarray(pts, dtype=np.float64), f=faces)
print(f"SDF_OFFSET ok voxel={voxel} offset={offset} faces={len(faces)} {time.time() - t0:.2f}s")

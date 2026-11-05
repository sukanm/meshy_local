"""Runs INSIDE Blender's bundled Python (must not import `pipeline`). Adds a
print base with Blender's Manifold boolean solver (falling back to the Exact
solver when Manifold declines the input); the result is validated by the caller.

Usage: blender -b --python base_boolean.py -- <in.npz> <out.npz> flatten <depth_mm>
       blender -b --python base_boolean.py -- <in.npz> <out.npz> plinth <thickness_mm> <embed_mm> <margin_mm>

Assumes Z-up millimetres (the print-prep stage orients before calling this).
  flatten: removes everything below (min_z + depth), leaving a flat contact patch.
  plinth:  unions an elliptical disc under the XY footprint; the model sinks
           `embed_mm` into the disc's top so the union genuinely fuses.
"""

import sys
import time

import bmesh
import bpy
import numpy as np

args = sys.argv[sys.argv.index("--") + 1 :]
in_path, out_path, mode = args[0], args[1], args[2]
t0 = time.time()

bpy.ops.wm.read_factory_settings(use_empty=True)

with np.load(in_path) as d:
    verts = d["v"].astype(np.float32)
    faces = d["f"].astype(np.int32)

# Bulk-load via foreach_set (from_pydata is pure-Python per element — far
# slower at the ~300K-face sizes this sees).
mesh = bpy.data.meshes.new("model")
mesh.vertices.add(len(verts))
mesh.vertices.foreach_set("co", verts.ravel())
mesh.loops.add(faces.size)
mesh.loops.foreach_set("vertex_index", faces.ravel())
mesh.polygons.add(len(faces))
mesh.polygons.foreach_set("loop_start", np.arange(0, faces.size, 3, dtype=np.int32))
mesh.update()
mesh.validate()
model = bpy.data.objects.new("model", mesh)
bpy.context.scene.collection.objects.link(model)

lo = verts.min(axis=0)
hi = verts.max(axis=0)
size = hi - lo
center = (lo + hi) / 2

cutter_mesh = bpy.data.meshes.new("cutter")
bm = bmesh.new()
if mode == "flatten":
    depth = float(args[3])
    # A box covering everything below the cut plane (with generous XY/Z margin).
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(cutter_mesh)
    cutter = bpy.data.objects.new("cutter", cutter_mesh)
    pad = float(max(size[0], size[1])) + 10.0
    cutter.scale = (pad * 2, pad * 2, 20.0)
    cutter.location = (float(center[0]), float(center[1]), float(lo[2]) + depth - 10.0)
    operation = "DIFFERENCE"
elif mode == "plinth":
    thickness, embed, margin = (float(x) for x in args[3:6])
    bmesh.ops.create_cone(
        bm, cap_ends=True, cap_tris=False, segments=128, radius1=0.5, radius2=0.5, depth=1.0
    )
    bm.to_mesh(cutter_mesh)
    cutter = bpy.data.objects.new("cutter", cutter_mesh)
    cutter.scale = (float(size[0]) + 2 * margin, float(size[1]) + 2 * margin, thickness)
    top = float(lo[2]) + embed
    cutter.location = (float(center[0]), float(center[1]), top - thickness / 2)
    operation = "UNION"
else:
    raise SystemExit(f"unknown mode {mode!r}")
bm.free()
bpy.context.scene.collection.objects.link(cutter)

mod = model.modifiers.new("base", "BOOLEAN")
mod.operation = operation
mod.object = cutter


def evaluate(solver):
    mod.solver = solver
    ev = model.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh()
    ev.calc_loop_triangles()
    return ev


# MANIFOLD is fast but silently returns the input unchanged when it doesn't
# consider the input strictly manifold (seen on detail-snapped meshes that
# are watertight by edge count), so fall back to the slower EXACT solver.
solver = "MANIFOLD"
evaluated = evaluate(solver)
if len(evaluated.loop_triangles) == len(faces) and len(evaluated.vertices) == len(verts):
    model.evaluated_get(bpy.context.evaluated_depsgraph_get()).to_mesh_clear()
    solver = "EXACT"
    evaluated = evaluate(solver)
n_v = len(evaluated.vertices)
n_t = len(evaluated.loop_triangles)
if n_t == 0:
    raise SystemExit("boolean produced an empty mesh (non-manifold input?)")
out_v = np.empty(n_v * 3, dtype=np.float32)
evaluated.vertices.foreach_get("co", out_v)
out_f = np.empty(n_t * 3, dtype=np.int32)
evaluated.loop_triangles.foreach_get("vertices", out_f)

np.savez(out_path, v=out_v.reshape(-1, 3).astype(np.float64), f=out_f.reshape(-1, 3).astype(np.int64))
print(f"BASE_BOOLEAN ok mode={mode} solver={solver} faces={n_t} {time.time() - t0:.2f}s")

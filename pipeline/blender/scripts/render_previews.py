"""Runs INSIDE Blender's bundled Python (must not import `pipeline`). Renders
orthographic Workbench previews of a Z-up STL — the actual exported file,
so the images show exactly what the slicer will get.

Usage: blender -b --python render_previews.py -- <model.stl> <out_dir> <resolution_px> <name:azimuth:elevation> ...

Workbench + matcap + cavity shading: fast (~0.03s/view after load) and
makes surface defects obvious — this is how the long "starburst" sliver
rods in job eeaae724ce2b were spotted.
"""

import math
import sys
from pathlib import Path

import bpy
import mathutils

args = sys.argv[sys.argv.index("--") + 1 :]
stl_path, out_dir, resolution = args[0], Path(args[1]), int(args[2])
views = [(n, float(a), float(e)) for n, a, e in (v.split(":") for v in args[3:])]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.stl_import(filepath=stl_path)
obj = bpy.context.selected_objects[0]

corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
lo = mathutils.Vector([min(c[i] for c in corners) for i in range(3)])
hi = mathutils.Vector([max(c[i] for c in corners) for i in range(3)])
center = (lo + hi) / 2
size = max(hi - lo)

scene = bpy.context.scene
scene.render.engine = "BLENDER_WORKBENCH"
scene.display.shading.light = "MATCAP"
scene.display.shading.show_cavity = True
scene.render.resolution_x = scene.render.resolution_y = resolution
scene.render.image_settings.file_format = "PNG"

cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
scene.collection.objects.link(cam)
scene.camera = cam
cam.data.type = "ORTHO"
cam.data.ortho_scale = size * 1.15
cam.data.clip_end = size * 10
distance = size * 2.0

out_dir.mkdir(parents=True, exist_ok=True)
for name, azimuth, elevation in views:
    a, e = math.radians(azimuth), math.radians(elevation)
    # Azimuth 0 looks from -Y toward +Y (Bambu Studio's default front view).
    offset = mathutils.Vector(
        (distance * math.sin(a) * math.cos(e), -distance * math.cos(a) * math.cos(e), distance * math.sin(e))
    )
    cam.location = center + offset
    cam.rotation_euler = (-offset).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(out_dir / f"preview_{name}.png")
    bpy.ops.render.render(write_still=True)
    print(f"PREVIEW {scene.render.filepath}")

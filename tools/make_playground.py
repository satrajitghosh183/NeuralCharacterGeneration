#!/usr/bin/env python3
"""Build a ready-to-play .blend around the avatar glb (run inside Blender, headless):

  /Applications/Blender.app/Contents/MacOS/Blender -b -P tools/make_playground.py -- \
      <avatar.glb> <out.blend>

Scene: avatar on a studio floor, lights + camera, armature shown as STICKS (no joint balls),
togglable CLOTHES (shirt + shorts built from the body mesh itself, so they keep the skin
weights and deform with the rig), and a 120-frame idle sway so pressing Space shows life.
"""
import sys

import bpy
import bmesh
from math import radians, sin

argv = sys.argv[sys.argv.index("--") + 1:]
GLB, OUT = argv[0], argv[1]

# ---- clean scene ------------------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
scn = bpy.context.scene

# ---- import avatar ----------------------------------------------------------------------------
bpy.ops.import_scene.gltf(filepath=GLB)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
body = next(o for o in bpy.data.objects if o.type == "MESH")
body.name = "Body"
arm.name = "Rig"
arm.data.display_type = "STICK"          # kills the joint-ball viewport clutter
arm.show_in_front = False

# ---- clothing from the body itself (keeps weights -> deforms with the rig) --------------------
zs = [ (body.matrix_world @ v.co).z for v in body.data.vertices ]
zmin, zmax = min(zs), max(zs)
H = zmax - zmin

def make_garment(name, lo_f, hi_f, color, thickness=0.006, max_x=None):
    dup = body.copy()
    dup.data = body.data.copy()
    dup.name = name
    scn.collection.objects.link(dup)
    bm = bmesh.new()
    bm.from_mesh(dup.data)
    lo, hi = zmin + lo_f * H, zmin + hi_f * H
    def outside(v):
        w = dup.matrix_world @ v.co
        if not (lo <= w.z <= hi):
            return True
        return max_x is not None and abs(w.x) > max_x  # trim sleeves (T-pose hands sit at shirt height)
    doomed = [v for v in bm.verts if outside(v)]
    bmesh.ops.delete(bm, geom=doomed, context="VERTS")
    # push the garment off the skin so the body never pokes through on curved areas
    for v in bm.verts:
        v.co += v.normal * 0.004
    bm.to_mesh(dup.data)
    bm.free()
    solid = dup.modifiers.new("Shell", "SOLIDIFY")
    solid.thickness = thickness
    solid.offset = 1.0
    subdiv = dup.modifiers.new("Smooth", "SUBSURF")   # smooths the ragged cut edges into cloth
    subdiv.levels = 2
    subdiv.render_levels = 2
    mat = bpy.data.materials.new(name + "Mat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.85
    if "Sheen Weight" in bsdf.inputs:                 # fabric response
        bsdf.inputs["Sheen Weight"].default_value = 0.4
    dup.data.materials.clear()
    dup.data.materials.append(mat)
    for p in dup.data.polygons:
        p.use_smooth = True
    return dup

shirt = make_garment("Shirt", 0.50, 0.80, (0.09, 0.12, 0.30), max_x=0.45)  # tee, sleeves above elbow
shorts = make_garment("Shorts", 0.34, 0.545, (0.13, 0.13, 0.14), thickness=0.008)  # overlaps shirt hem

# ---- floor / lights / camera / world -----------------------------------------------------------
bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, zmin))
floor = bpy.context.object
floor.name = "Floor"
fmat = bpy.data.materials.new("FloorMat")
fmat.use_nodes = True
nt = fmat.node_tree
checker = nt.nodes.new("ShaderNodeTexChecker")
checker.inputs["Scale"].default_value = 24.0
checker.inputs["Color1"].default_value = (0.55, 0.55, 0.57, 1)
checker.inputs["Color2"].default_value = (0.42, 0.42, 0.44, 1)
nt.links.new(checker.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
floor.data.materials.append(fmat)

sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
sun.data.energy = 3.0
sun.rotation_euler = (radians(50), 0, radians(30))
scn.collection.objects.link(sun)
key = bpy.data.objects.new("Key", bpy.data.lights.new("Key", "AREA"))
key.data.energy = 400
key.data.size = 3
key.location = (2.5, -2.5, zmin + 0.75 * H)
key.rotation_euler = (radians(60), 0, radians(45))
scn.collection.objects.link(key)

cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
cam.location = (0, -4.2, zmin + 0.62 * H)
cam.rotation_euler = (radians(87), 0, 0)
scn.collection.objects.link(cam)
scn.camera = cam

world = bpy.data.worlds.new("World")
scn.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.35, 0.4, 0.5, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6

# ---- 120-frame idle sway (subtle breath + arm/head motion; loops) ------------------------------
scn.frame_start, scn.frame_end = 1, 120
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")

def sway(bone_name, axis, amp_deg, phase=0.0):
    pb = arm.pose.bones.get(bone_name)
    if pb is None:
        return
    pb.rotation_mode = "XYZ"
    for f in range(1, 121, 10):
        t = (f - 1) / 120.0
        rot = [0.0, 0.0, 0.0]
        rot[axis] = radians(amp_deg) * sin(2 * 3.14159 * (t + phase))
        pb.rotation_euler = rot
        pb.keyframe_insert("rotation_euler", frame=f)

sway("spine2", 0, 2.0)             # breathing lean
sway("head", 2, 4.0, phase=0.25)   # slow look-around
sway("left_shoulder", 1, 3.0, phase=0.5)
sway("right_shoulder", 1, -3.0, phase=0.5)
bpy.ops.object.mode_set(mode="OBJECT")

# ---- viewport niceties + save ------------------------------------------------------------------
scn.render.engine = "BLENDER_EEVEE"
try:
    bpy.ops.file.pack_all()
except Exception:
    pass
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print("PLAYGROUND SAVED:", OUT)

# quick preview render for the eye-check
scn.render.filepath = OUT.replace(".blend", "_preview.png")
scn.render.resolution_x, scn.render.resolution_y = 900, 700
bpy.ops.render.render(write_still=True)
print("PREVIEW:", scn.render.filepath)

#!/usr/bin/env python3
"""Render the REAL textured mesh (not the splat preview) so the baked face texture is shown sharply,
the way it looks in an engine. DEV/visualization tool. Loads the verts/faces/uvcoords/uvfaces npy +
a UV texture png, unwelds the SMPL-X UV seams (uvfaces != faces), and rasterizes via pyrender (EGL,
headless). Frames tight on the head.

  PYOPENGL_PLATFORM=egl python tools/render_textured.py --dir runs/face_XXXX --prefix myFINAL2 \
      --tex myFINAL2_albedo_baked_uv.png --out /tmp/model
"""
import argparse
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
import trimesh
import pyrender
from PIL import Image


def look_at(eye, target, up=(0, 1, 0)):
    eye = np.asarray(eye, float); target = np.asarray(target, float); up = np.asarray(up, float)
    f = target - eye; f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4)
    m[:3, 0] = r; m[:3, 1] = u; m[:3, 2] = -f; m[:3, 3] = eye
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--tex", required=True, help="texture png filename inside --dir")
    ap.add_argument("--out", default="/tmp/model")
    ap.add_argument("--res", type=int, default=1024)
    args = ap.parse_args()
    d = args.dir

    verts = np.load(f"{d}/{args.prefix}_verts.npy").astype(np.float32)      # [V,3]
    faces = np.load(f"{d}/{args.prefix}_faces.npy").astype(np.int64)        # [F,3]
    uv = np.load(f"{d}/{args.prefix}_uvcoords.npy").astype(np.float32)      # [VT,2]
    uvfaces = np.load(f"{d}/{args.prefix}_uvfaces.npy").astype(np.int64)    # [F,3]
    tex = Image.open(f"{d}/{args.tex}").convert("RGB")

    # Unweld so each triangle corner carries its own UV (SMPL-X uvfaces != faces).
    vf = verts[faces.reshape(-1)]                                           # [F*3,3]
    uvf = uv[uvfaces.reshape(-1)].copy()                                    # [F*3,2]
    uvf[:, 1] = 1.0 - uvf[:, 1]                                             # glTF/OpenGL V flip
    nf = np.arange(len(vf)).reshape(-1, 3)
    vis = trimesh.visual.texture.TextureVisuals(uv=uvf, image=tex)
    mesh = trimesh.Trimesh(vertices=vf, faces=nf, visual=vis, process=False)

    # Frame on the head: top 18% of the body in Y (the head), centroid + radius from its extent.
    y = verts[:, 1]
    head = verts[y > np.quantile(y, 0.82)]
    center = head.mean(0)
    rad = float(np.linalg.norm(head - center, axis=1).max()) * 2.4

    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.45, 0.45, 0.45])
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))
    cam = pyrender.PerspectiveCamera(yfov=np.pi / 5.0)
    key = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    r = pyrender.OffscreenRenderer(args.res, args.res)

    # Auto-pick the facing direction: render +Z and -Z, keep the brighter (the face side has texture).
    best = None
    for name, az in [("front", 0.0), ("left", -35.0), ("right", 35.0)]:
        a = np.radians(az)
        for zdir in ([1.0, -1.0] if name == "front" else [best_z]):
            eye = center + np.array([np.sin(a) * rad * zdir, 0.0, np.cos(a) * rad * zdir])
            pose = look_at(eye, center)
            scene_cam = scene.add(cam, pose=pose)
            scene_l = scene.add(key, pose=pose)
            color, _ = r.render(scene)
            scene.remove_node(scene_cam); scene.remove_node(scene_l)
            if name == "front":
                bright = color[..., :3].mean()
                if best is None or bright > best[0]:
                    best = (bright, color.copy(), zdir)
        if name == "front":
            best_z = best[2]
            Image.fromarray(best[1]).save(f"{args.out}_front.png")
            print(f"[render] front (zdir={best_z}) -> {args.out}_front.png")
        else:
            Image.fromarray(color).save(f"{args.out}_{name}.png")
            print(f"[render] {name} -> {args.out}_{name}.png")
    r.delete()


if __name__ == "__main__":
    main()

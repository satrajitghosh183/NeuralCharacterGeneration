#!/usr/bin/env python3
"""Pure-PyTorch software rasterizer (no OpenGL) to show the REAL textured mesh sharply. Loads
verts/faces/uvcoords/uvfaces npy + a UV texture png, unwelds SMPL-X UV seams, orthographically
rasterizes the head with a z-buffer + bilinear texture sampling + simple Lambertian shading. DEV tool.

  python tools/raster_textured.py --dir runs/face_XXXX --prefix myFINAL2 \
      --tex myFINAL2_albedo_baked_uv.png --out /tmp/model --res 900
"""
import argparse
import numpy as np
import torch
from PIL import Image


def sample_tex(tex, uv):  # tex [Ht,Wt,3], uv [N,2] in [0,1] -> [N,3] bilinear
    Ht, Wt = tex.shape[0], tex.shape[1]
    x = uv[:, 0].clamp(0, 1) * (Wt - 1)
    y = (1.0 - uv[:, 1].clamp(0, 1)) * (Ht - 1)  # V flip
    x0 = x.floor().long(); y0 = y.floor().long()
    x1 = (x0 + 1).clamp(0, Wt - 1); y1 = (y0 + 1).clamp(0, Ht - 1)
    wx = (x - x0).unsqueeze(1); wy = (y - y0).unsqueeze(1)
    a = tex[y0, x0]; b = tex[y0, x1]; c = tex[y1, x0]; d = tex[y1, x1]
    return (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy


def render(verts, faces, uv, uvfaces, tex, res, zdir, dev):
    V = torch.tensor(verts, device=dev, dtype=torch.float32)
    F = torch.tensor(faces, device=dev, dtype=torch.long)
    UV = torch.tensor(uv, device=dev, dtype=torch.float32)
    UVF = torch.tensor(uvfaces, device=dev, dtype=torch.long)
    TEX = torch.tensor(np.asarray(tex, np.float32) / 255.0, device=dev)

    y = V[:, 1]
    headv = V[y > torch.quantile(y, 0.55)]                  # whole head
    yh0 = float(headv[:, 1].min()); yh1 = float(headv[:, 1].max())
    cx = float(headv[:, 0].mean())
    cy = yh0 + 0.46 * (yh1 - yh0)                           # center vertically on the face
    half = (yh1 - yh0) * 0.60                               # frame the head height
    sx = (V[:, 0] - (cx - half)) / (2 * half) * res
    sy = (1.0 - (V[:, 1] - (cy - half)) / (2 * half)) * res
    depth = zdir * V[:, 2]  # larger = nearer

    # keep head triangles only (all 3 verts above the neck) to cut work
    keep = (y[F] > torch.quantile(y, 0.55)).all(1)
    F = F[keep]; UVF = UVF[keep]
    a, b, cc = F[:, 0], F[:, 1], F[:, 2]
    pa = torch.stack([sx[a], sy[a]], 1); pb = torch.stack([sx[b], sy[b]], 1)
    pc = torch.stack([sx[cc], sy[cc]], 1)
    za, zb, zc = depth[a], depth[b], depth[c if False else cc]
    uva = UV[UVF[:, 0]]; uvb = UV[UVF[:, 1]]; uvc = UV[UVF[:, 2]]
    # face normals (for shading) in world space
    n = torch.cross(V[b] - V[a], V[cc] - V[a], dim=1)
    n = n / n.norm(dim=1, keepdim=True).clamp_min(1e-8)
    facing = zdir * n[:, 2]                                 # >0 => triangle faces the camera
    shade = (0.5 + 0.5 * facing.clamp(0, 1))               # frontal Lambertian-ish

    img = torch.zeros(res, res, 3, device=dev)
    zbuf = torch.full((res, res), -1e9, device=dev)
    area = (pb[:, 0] - pa[:, 0]) * (pc[:, 1] - pa[:, 1]) - (pb[:, 1] - pa[:, 1]) * (pc[:, 0] - pa[:, 0])
    for i in range(F.shape[0]):
        A = area[i]
        if abs(A.item()) < 1e-6 or facing[i].item() <= -0.25:  # backface cull
            continue
        x0 = int(max(0, torch.floor(torch.min(torch.stack([pa[i, 0], pb[i, 0], pc[i, 0]]))).item()))
        x1 = int(min(res - 1, torch.ceil(torch.max(torch.stack([pa[i, 0], pb[i, 0], pc[i, 0]]))).item()))
        y0 = int(max(0, torch.floor(torch.min(torch.stack([pa[i, 1], pb[i, 1], pc[i, 1]]))).item()))
        y1 = int(min(res - 1, torch.ceil(torch.max(torch.stack([pa[i, 1], pb[i, 1], pc[i, 1]]))).item()))
        if x1 < x0 or y1 < y0:
            continue
        ys, xs = torch.meshgrid(torch.arange(y0, y1 + 1, device=dev),
                                torch.arange(x0, x1 + 1, device=dev), indexing="ij")
        px = xs.flatten().float() + 0.5; py = ys.flatten().float() + 0.5
        w0 = ((pb[i, 0] - px) * (pc[i, 1] - py) - (pb[i, 1] - py) * (pc[i, 0] - px)) / A
        w1 = ((pc[i, 0] - px) * (pa[i, 1] - py) - (pc[i, 1] - py) * (pa[i, 0] - px)) / A
        w2 = 1 - w0 - w1
        m = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if m.sum() == 0:
            continue
        w0, w1, w2 = w0[m], w1[m], w2[m]
        gx, gy = xs.flatten()[m], ys.flatten()[m]
        z = w0 * za[i] + w1 * zb[i] + w2 * zc[i]
        cur = zbuf[gy, gx]
        upd = z > cur
        if upd.sum() == 0:
            continue
        gx, gy, w0, w1, w2, z = gx[upd], gy[upd], w0[upd], w1[upd], w2[upd], z[upd]
        uvp = w0.unsqueeze(1) * uva[i] + w1.unsqueeze(1) * uvb[i] + w2.unsqueeze(1) * uvc[i]
        col = sample_tex(TEX, uvp) * shade[i]
        zbuf[gy, gx] = z
        img[gy, gx] = col
    return (img.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--prefix", required=True)
    ap.add_argument("--tex", required=True); ap.add_argument("--out", default="/tmp/model")
    ap.add_argument("--res", type=int, default=900)
    args = ap.parse_args()
    d = args.dir
    verts = np.load(f"{d}/{args.prefix}_verts.npy").astype(np.float32)
    faces = np.load(f"{d}/{args.prefix}_faces.npy").astype(np.int64)
    uv = np.load(f"{d}/{args.prefix}_uvcoords.npy").astype(np.float32)
    uvfaces = np.load(f"{d}/{args.prefix}_uvfaces.npy").astype(np.int64)
    tex = Image.open(f"{d}/{args.tex}").convert("RGB")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for zdir, nm in ((1.0, "zpos"), (-1.0, "zneg")):
        im = render(verts, faces, uv, uvfaces, tex, args.res, zdir, dev)
        Image.fromarray(im).save(f"{args.out}_{nm}.png")
        print(f"[raster] {nm} (cover={float((im.mean(2)>8).mean()):.2f}) -> {args.out}_{nm}.png")


if __name__ == "__main__":
    main()

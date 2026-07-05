#!/usr/bin/env python3
"""Build a REAL iris tile from the subject's own photos (DEV-side data prep, like the exporters).

Detects the face + 5-point landmarks (retinaface via facexlib) on the sharpest frontal album
photos, crops both eye regions, extracts the iris radial color profile from the real pixels, and
composes a square eye tile:  pupil core -> the subject's actual iris colors -> sclera rim
(+ catchlight). The C++ eyeball geometry maps sphere verts radially into this tile.

  python tools/make_iris.py --album data/me_clean_album --face-src face_src --out models/face/iris_tile.png
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageFilter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--album", required=True)
    ap.add_argument("--face-src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tile", type=int, default=96)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    from facexlib.detection import init_detection_model
    det = init_detection_model("retinaface_resnet50", half=False, device="cpu")

    # score photos: detection confidence x face size x sharpness
    cands = []
    for f in sorted(glob.glob(os.path.join(args.album, "*.jpg"))):
        im = Image.open(f).convert("RGB")
        im.thumbnail((1200, 1200))
        arr = np.asarray(im)[:, :, ::-1].copy()  # BGR
        with torch.no_grad():
            bboxes = det.detect_faces(arr, 0.8)
        if len(bboxes) == 0:
            continue
        b = max(bboxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        area = (b[2] - b[0]) * (b[3] - b[1])
        gray = np.asarray(im.convert("L"), dtype=np.float32)
        sharp = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
        cands.append((float(b[4]) * area * sharp, f, b))
    cands.sort(reverse=True)
    print(f"[iris] {len(cands)} photos with faces; using top {args.top_k}")

    # collect eye crops (landmarks 5-14 of retinaface bbox row: x1,y1,x2,y2,score, l0x,l0y ... l4y;
    # landmark 0 = left eye, 1 = right eye)
    crops = []
    for _, f, b in cands[:args.top_k]:
        im = Image.open(f).convert("RGB")
        im.thumbnail((1200, 1200))
        eye_w = max(10, int((b[2] - b[0]) * 0.18))
        for li in (0, 1):
            ex, ey = b[5 + 2 * li], b[6 + 2 * li]
            crop = im.crop((ex - eye_w, ey - eye_w, ex + eye_w, ey + eye_w)).resize((64, 64),
                                                                                    Image.LANCZOS)
            a = np.asarray(crop, dtype=np.float32) / 255.0
            if li == 1:
                a = a[:, ::-1]  # mirror right eye
            crops.append(a)
    if not crops:
        raise SystemExit("[iris] no eye crops found")
    avg = np.mean(crops, 0)  # [64,64,3] the subject's average eye

    # radial iris profile around the crop center (the iris is the darker disc around the center;
    # sample median color per radius ring)
    cy = cx = 32
    yy, xx = np.mgrid[0:64, 0:64]
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    prof = []
    for r0 in range(0, 22):
        ring = avg[(rr >= r0) & (rr < r0 + 1)]
        prof.append(np.median(ring, 0) if len(ring) else prof[-1])
    prof = np.array(prof)  # [22,3] radial colors: pupil-ish core -> iris -> lids/sclera mix

    # compose the tile: r in [0,0.16] pupil (near black), (0.16,0.62] REAL iris profile,
    # (0.62,0.70] limbal ring (darkened), (0.70,1] sclera
    S = args.tile
    yy, xx = np.mgrid[0:S, 0:S].astype(np.float32)
    r = np.sqrt((yy - S / 2) ** 2 + (xx - S / 2) ** 2) / (S / 2)
    tile = np.zeros((S, S, 3), np.float32)
    sclera = np.array([0.93, 0.90, 0.88])
    for i in range(S):
        for j in range(S):
            q = r[i, j]
            if q <= 0.10:
                tile[i, j] = (0.03, 0.025, 0.025)
            elif q <= 0.35:
                t = (q - 0.10) / 0.25
                k = 4 + t * 14  # map into the measured 4..18px iris band of the 22-ring profile
                k0 = int(np.clip(k, 0, 20))
                fr = k - k0
                tile[i, j] = prof[k0] * (1 - fr) + prof[min(k0 + 1, 21)] * fr
            elif q <= 0.42:
                t = (q - 0.35) / 0.07
                tile[i, j] = (prof[18] * (1 - t) + sclera * t) * (0.75 + 0.25 * t)  # limbal ring
            else:
                shade = 1.0 - 0.10 * max(0.0, (q - 0.85)) / 0.15  # slight corner shading
                tile[i, j] = sclera * shade
    # catchlight
    ci, cj = int(S * 0.44), int(S * 0.46)
    for di in range(-3, 4):
        for dj in range(-3, 4):
            if di * di + dj * dj <= 9:
                tile[ci + di, cj + dj] = np.minimum(tile[ci + di, cj + dj] + 0.75, 1.0)
    img = Image.fromarray((np.clip(tile, 0, 1) * 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    img.save(args.out)
    iris_mean = prof[8:16].mean(0)
    print(f"[iris] wrote {args.out}  (subject iris tone ~ {iris_mean.round(3)})")


if __name__ == "__main__":
    main()

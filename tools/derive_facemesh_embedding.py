#!/usr/bin/env python3
"""Derive the FaceMesh-468 -> SMPL-X mesh embedding (Phase B §M2), data-driven, no gated assets.
On a clean frontal reference photo: NLF projects the SMPL-X mesh to 2D (vertices2d), and FaceMesh
gives 468 independent 2D facial points. For each FaceMesh point we find the SMPL-X triangle that
contains it (2D point-in-triangle) and its barycentric weights. The result (assoc[468], bary[468,3])
is a FIXED topological correspondence reused for every photo.  DEV-ONLY.

  python tools/derive_facemesh_embedding.py --ref data/me_album/p14.jpg \
     --nlf models/nlf_l_multi.torchscript --detector data/face/detector.torchscript \
     --facemesh data/face/facemesh.torchscript --smplx data/smplx_neutral.safetensors \
     --out data/face/facemesh_smplx_embed.npz
"""
import argparse
import numpy as np
import torch
import torchvision  # registers nms for the scripted detector  # noqa: F401
from PIL import Image
from safetensors.numpy import load_file


def barycentric(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p[None] - a            # [F,2] each
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1 - v - w
    return np.stack([u, v, w], 1)                     # [F,3]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--nlf", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--facemesh", required=True)
    ap.add_argument("--smplx", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    im = np.asarray(Image.open(args.ref).convert("RGB"))
    H, W = im.shape[:2]
    arr01 = torch.from_numpy(im).float().permute(2, 0, 1) / 255

    # NLF -> SMPL-X vertices2d (the mesh projection in image pixels).
    nlf = torch.jit.load(args.nlf).eval().to(dev)
    fr = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).contiguous().to(dev)
    with torch.no_grad():
        out = nlf.detect_smpl_batched(fr, model_name="smplx")
    v2d = out["vertices2d"][0][0].float().cpu().numpy()   # [V,2]
    faces = load_file(args.smplx)["faces"].astype(np.int64)  # [F,3]

    # FaceMesh -> 468 dense points (image pixels), via detector top box.
    det = torch.jit.load(args.detector).eval()
    fm = torch.jit.load(args.facemesh).eval()
    with torch.no_grad():
        boxes = det(arr01)
        assert boxes.shape[0] > 0, "no face detected in reference"
        b = boxes[0]
        x0, y0, x1, y1 = [float(t) for t in b[:4]]
        crop = arr01[:, int(max(0, y0)):int(min(H, y1)), int(max(0, x0)):int(min(W, x1))]
        crop = torch.nn.functional.interpolate(crop.unsqueeze(0), size=(192, 192), mode="bilinear",
                                               align_corners=False)
        uv, _conf = fm(crop)
    uv = uv.cpu().numpy()                                 # [468,2] in [0,1]
    pts = np.stack([uv[:, 0] * (x1 - x0) + x0, uv[:, 1] * (y1 - y0) + y0], 1)  # [468,2] image px

    # Candidate faces: those whose projected centroid is in the FaceMesh bbox (+margin) — the face.
    fc = v2d[faces].mean(1)                                # [F,2] face centroids
    lo = pts.min(0) - 20; hi = pts.max(0) + 20
    cand = np.where((fc[:, 0] > lo[0]) & (fc[:, 0] < hi[0]) & (fc[:, 1] > lo[1]) & (fc[:, 1] < hi[1]))[0]
    fa = v2d[faces[cand, 0]]; fb = v2d[faces[cand, 1]]; fcc = v2d[faces[cand, 2]]

    assoc = np.zeros(468, np.int64); bary = np.zeros((468, 3), np.float32); inside = 0
    for k in range(468):
        bc = barycentric(pts[k], fa, fb, fcc)             # [Fc,3]
        ok = (bc >= -0.02).all(1) & (bc <= 1.02).all(1)
        if ok.any():
            j = np.where(ok)[0]
            # among containing faces pick the most central (max min-bary).
            best = j[np.argmax(bc[j].min(1))]; inside += 1
        else:
            best = np.argmin(np.abs(bc).max(1))           # nearest triangle
        assoc[k] = cand[best]; bary[k] = np.clip(bc[best], 0, 1)
        bary[k] /= bary[k].sum()
    print(f"derived embedding: {inside}/468 points inside a SMPL-X face, {len(cand)} candidate faces")
    np.savez(args.out, assoc=assoc, bary=bary)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

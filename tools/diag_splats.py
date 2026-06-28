#!/usr/bin/env python3
"""Free-splat layer gate measurements (GATE1 density/opacity, GATE3 floaters) from a 3DGS .ply."""
import argparse, re, numpy as np


def parse_ply(path):
    d = open(path, "rb").read()
    he = d.index(b"end_header\n") + len(b"end_header\n")
    hdr = d[:he].decode("latin1")
    N = int(re.search(r"element vertex (\d+)", hdr).group(1))
    props = re.findall(r"property float (\w+)", hdr)
    body = np.frombuffer(d[he:he + N * len(props) * 4], dtype=np.float32).reshape(N, len(props))
    return N, props, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--verts", required=True, help="Layer-1 mesh verts npy (for face region + floater dist)")
    ap.add_argument("--head-q", type=float, default=0.86)
    args = ap.parse_args()
    N, props, body = parse_ply(args.ply)
    xyz = body[:, :3]
    op = body[:, props.index("opacity")]
    v = np.load(args.verts).astype(np.float32)

    y = v[:, 1]
    head_v = v[y > np.quantile(y, args.head_q)]
    yc = head_v[:, 1].min()
    face_splats = xyz[xyz[:, 1] > yc]
    # opacity -> probability (stored as logit in Inria convention)
    prob = 1.0 / (1.0 + np.exp(-op))

    # floaters: opaque splats far from ANY mesh vertex (subsample mesh for speed)
    vs = v[np.random.RandomState(0).choice(len(v), min(4000, len(v)), replace=False)]
    # nearest mesh-vertex distance for opaque splats
    opaque = xyz[prob > 0.5]
    if len(opaque) > 20000:
        opaque = opaque[np.random.RandomState(1).choice(len(opaque), 20000, replace=False)]
    d2 = ((opaque[:, None, :] - vs[None, :, :]) ** 2).sum(2).min(1) ** 0.5
    # scale: head ~0.22m → mm
    head_h = float(y.max() - np.quantile(y, args.head_q))
    mm = 0.22 / max(head_h, 1e-6) * 1000
    floater_frac = float((d2 * mm / 1000 > 0.03).mean())  # >3cm from any vert

    print(f"total_N = {N}")
    print(f"is_round_10475 = {N == 10475}")
    print(f"opacity_std = {op.std():.5f}  (constant==densify_didnt_run)")
    print(f"opacity_unique = {np.unique(np.round(op,3)).size}")
    print(f"face_region_splats = {len(face_splats)}")
    print(f"prob_mean = {prob.mean():.3f}  prob_min = {prob.min():.3f}")
    print(f"floater_frac_gt3cm = {floater_frac:.4f}  (opaque splats far from mesh)")


if __name__ == "__main__":
    main()

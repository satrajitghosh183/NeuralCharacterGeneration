#!/usr/bin/env python3
"""Build a per-pixel CONFIDENCE GATE for differential-diffusion refinement of a face base render.

The gate is the change-map for differential diffusion: HIGH (white) = low confidence => let the prior
COMPLETE (high strength); LOW (black) = high confidence => REFINE only (low strength), identity held.

Confidence proxy for a frontal-dominant capture (faithful, not arbitrary):
  - FRONTALITY / centrality: the recovered identity is best observed in the central frontal face;
    confidence falls off toward the periphery (hairline, ears, jaw edges) which were rarely seen.
  - SEAM/CRACK penalty: the recovered albedo's UV-seam cracks are exactly where the multi-view solve
    disagreed (low confidence). They show as high-frequency edges; we mark them low-confidence so the
    prior repaints them instead of following them.
Outside the silhouette (alpha==0) the gate is 0 (don't touch the black background).

Output: a grayscale PNG, same size as the base. (A rigorous per-view o(v) render from the C++ solve
would replace the frontality term; this is the Python stand-in that uses the same principle.)
"""
import argparse

import numpy as np
from PIL import Image, ImageFilter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--center-sigma", type=float, default=0.42, help="centrality falloff (frac of size)")
    ap.add_argument("--seam-weight", type=float, default=0.6)
    ap.add_argument("--floor", type=float, default=0.12, help="min change even in confident regions (refine)")
    ap.add_argument("--ceil", type=float, default=0.95)
    args = ap.parse_args()

    img = Image.open(args.base).convert("RGB")
    W, H = img.size
    arr = np.asarray(img).astype(np.float32) / 255.0
    lum = arr.mean(2)
    fg = (lum > 0.04).astype(np.float32)  # silhouette (black bg => 0)

    # centrality (frontal-observed confidence): gaussian about the foreground centroid.
    ys, xs = np.nonzero(fg)
    cx, cy = (xs.mean(), ys.mean()) if len(xs) else (W / 2, H / 2)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    s = args.center_sigma * max(W, H)
    central = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s)))  # 1 center -> 0 edge

    # seam/crack response: high-pass of luminance (UV-seam cracks = low confidence).
    hp = np.asarray(Image.fromarray((lum * 255).astype(np.uint8)).filter(
        ImageFilter.FIND_EDGES)).astype(np.float32) / 255.0
    hp = np.asarray(Image.fromarray((hp * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(2))).astype(np.float32) / 255.0
    hp = hp / max(hp.max(), 1e-6)

    confidence = central * (1.0 - args.seam_weight * hp)  # high center & smooth -> confident
    confidence = np.clip(confidence, 0.0, 1.0)
    change = 1.0 - confidence  # differential-diffusion map: high => complete, low => refine
    change = args.floor + (args.ceil - args.floor) * change
    change = change * fg  # never touch the background

    Image.fromarray((np.clip(change, 0, 1) * 255).astype(np.uint8)).save(args.out)
    print(f"[make_gate] wrote {args.out}  (mean change inside fg = "
          f"{(change[fg > 0].mean() if fg.sum() else 0):.3f})", flush=True)


if __name__ == "__main__":
    main()

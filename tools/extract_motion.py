#!/usr/bin/env python3
"""Extract a SMPL-X motion sequence from a folder of video frames, via NLF (batched).

DEV-ONLY (workstation / H100). The foundation of the motion system (docs/plan.md): turns a video
of a person moving into per-frame SMPL-X pose — the data a style model trains on, and the simplest
"drive the avatar with this motion" path. Frames must be same-size (ffmpeg output is).

    python tools/extract_motion.py --model models/nlf_l_multi.torchscript \
        --frames frames_dir --out motion.npy [--batch 8 --detection 0 --model-name smplx]

Emits motion.npy of shape [T, J, 3] (axis-angle local joint rotations, J=55 for SMPL-X).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--frames", required=True, help="directory of same-size frame images")
    ap.add_argument("--out", default="motion.npy")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--detection", type=int, default=0)
    ap.add_argument("--model-name", default="smplx")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torchvision  # noqa: F401  -- registers torchvision::nms used by NLF's detector
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(args.model).to(device).eval()

    files = sorted(
        f for ext in ("*.jpg", "*.jpeg", "*.png")
        for f in glob.glob(os.path.join(args.frames, ext)))
    if not files:
        print("no frames found in", args.frames)
        return 1
    print(f"extracting motion from {len(files)} frames…")

    def load(p):
        return torch.from_numpy(np.asarray(Image.open(p).convert("RGB"))).permute(2, 0, 1)

    poses = []
    J = None
    with torch.inference_mode():
        for s in range(0, len(files), args.batch):
            chunk = files[s:s + args.batch]
            imgs = torch.stack([load(f) for f in chunk]).to(device)  # [B,3,H,W] uint8
            pred = getattr(model, "detect_smpl_batched")(imgs, model_name=args.model_name)
            for k in range(len(chunk)):
                pk = pred["pose"][k]  # [num_det, J*3]
                if pk.shape[0] == 0:
                    poses.append(None)  # no detection this frame; fill later
                else:
                    d = min(args.detection, pk.shape[0] - 1)
                    p = pk[d].reshape(-1, 3).float().cpu().numpy()
                    J = p.shape[0]
                    poses.append(p)
            print(f"  {min(s + args.batch, len(files))}/{len(files)}")

    J = J or 55
    # Fill missing-detection frames by holding the previous valid pose (or zeros at the start).
    last = np.zeros((J, 3), dtype=np.float32)
    motion = []
    for p in poses:
        if p is not None:
            last = p
        motion.append(last)
    motion = np.stack(motion).astype(np.float32)  # [T,J,3]
    np.save(args.out, motion)
    print(f"wrote {args.out}  shape={motion.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

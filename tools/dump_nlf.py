#!/usr/bin/env python3
"""Dump the released NLF TorchScript model's I/O structure + a golden for the C++ port.

DEV-ONLY (workstation / H100). Requires the released model from github.com/isarandi/nlf
(e.g. nlf_l_multi.torchscript) and torch + torchvision. Run this once after downloading the
checkpoint; paste manifest.json back so the C++ Nlf::predict output parsing can be confirmed
and locked with tests/golden/test_golden_nlf.cpp.

    python tools/dump_nlf.py --model models/nlf_l_multi.torchscript \
                             --image data/photos/me.jpg \
                             --out data/golden/nlf

Emits, under --out:
  manifest.json   - method name, input shape/dtype, and every output key's type/shape
  pose.npy betas.npy trans.npy   - the consumed params (image 0, detection 0)
  input_u8.npy    - the exact uint8 [1,3,H,W] tensor fed to the model
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def describe(v):
    import torch

    if torch.is_tensor(v):
        return {"type": "Tensor", "shape": list(v.shape), "dtype": str(v.dtype)}
    if isinstance(v, (list, tuple)):
        return {"type": "list", "len": len(v),
                "elem0": describe(v[0]) if len(v) else None}
    if isinstance(v, dict):
        return {"type": "dict", "keys": {k: describe(x) for k, x in v.items()}}
    return {"type": type(v).__name__}


def first(v):
    # Unwrap the per-image list NLF returns, then take detection 0.
    if isinstance(v, (list, tuple)):
        v = v[0]
    return v


def load_image_u8_chw(path):
    """Read an image as a uint8 RGB [3,H,W] tensor without requiring torchvision."""
    import numpy as np
    import torch

    arr = None
    try:
        from PIL import Image
        arr = np.array(Image.open(path).convert("RGB"))            # HWC uint8
    except Exception:
        try:
            import imageio.v2 as imageio
            arr = np.asarray(imageio.imread(path))[..., :3]
        except Exception:
            import torchvision  # last resort
            return torchvision.io.read_image(path)[:3]
    return torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).contiguous()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to nlf_l_multi.torchscript")
    ap.add_argument("--image", required=True, help="a test image (any in-the-wild photo)")
    ap.add_argument("--out", default="data/golden/nlf")
    ap.add_argument("--method", default="detect_smpl_batched")
    args = ap.parse_args()

    import numpy as np
    import torch

    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(args.model).to(device).eval()

    image = load_image_u8_chw(args.image).to(device)  # uint8 [3,H,W] RGB
    frames = image.unsqueeze(0)                        # [1,3,H,W]

    with torch.inference_mode():
        pred = getattr(model, args.method)(frames)

    manifest = {
        "method": args.method,
        "input_shape": list(frames.shape),
        "input_dtype": str(frames.dtype),
        "output_type": describe(pred),
    }

    # Save the params the C++ side consumes, for an exact parity check.
    saved = {}
    if isinstance(pred, dict):
        for k in ("pose", "betas", "trans"):
            if k in pred:
                arr = first(pred[k]).detach().to("cpu").float().numpy()
                np.save(os.path.join(args.out, f"{k}.npy"), arr)
                saved[k] = list(arr.shape)
    manifest["saved_param_shapes"] = saved
    np.save(os.path.join(args.out, "input_u8.npy"), frames.detach().cpu().numpy())

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2))
    print(f"\nwrote golden -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

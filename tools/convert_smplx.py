#!/usr/bin/env python3
"""Convert the official SMPL-X model (.npz / .pkl) to the safetensors layout ncg-body expects.

DEV-ONLY (workstation). Requires the license-gated SMPL-X model file from
https://smpl-x.is.tue.mpg.de/ . Emits the six keys SmplxModel::load reads:
v_template, shapedirs, posedirs, J_regressor, lbs_weights, parents.

    python tools/convert_smplx.py --in SMPLX_NEUTRAL.npz --out data/weights/smplx_neutral.safetensors
                                  [--num-betas 10]
"""
from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np
from safetensors.numpy import save_file


def load_model(path: str) -> dict:
    if path.endswith(".npz"):
        return dict(np.load(path, allow_pickle=True))
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def to_np(x) -> np.ndarray:
    # SMPL-X stores some arrays as scipy sparse / chumpy; coerce to dense float32.
    if hasattr(x, "todense"):
        x = np.asarray(x.todense())
    return np.asarray(x, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-betas", type=int, default=10)
    args = ap.parse_args()

    m = load_model(args.inp)

    v_template = to_np(m["v_template"])                 # [V,3]
    shapedirs = to_np(m["shapedirs"])[:, :, : args.num_betas]  # [V,3,nbetas]
    posedirs = to_np(m["posedirs"])                     # [V,3,9*(J-1)] or [V*3, 9*(J-1)]
    if posedirs.ndim == 2:
        posedirs = posedirs.reshape(v_template.shape[0], 3, -1)
    J_regressor = to_np(m["J_regressor"])               # [J,V]
    lbs_weights = to_np(m["weights"])                   # [V,J]
    kintree = np.asarray(m["kintree_table"])            # [2,J]
    parents = kintree[0].astype(np.int64)
    parents[0] = 0                                      # root parent (ignored by forward)

    tensors = {
        "v_template": v_template,
        "shapedirs": np.ascontiguousarray(shapedirs, dtype=np.float32),
        "posedirs": np.ascontiguousarray(posedirs, dtype=np.float32),
        "J_regressor": J_regressor,
        "lbs_weights": lbs_weights,
        "parents": parents,
    }
    if "f" in m:  # mesh triangles -> needed for vertex normals (relighting) + glTF export
        tensors["faces"] = np.ascontiguousarray(np.asarray(m["f"]), dtype=np.int64)
    for k, v in tensors.items():
        print(f"  {k}: {v.shape} {v.dtype}")
    save_file(tensors, args.out, metadata={"__ncg__": f"SMPL-X from {args.inp}"})
    print(f"wrote SMPL-X -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

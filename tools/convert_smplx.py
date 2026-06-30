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
    # MUST be C-contiguous: the npz J_regressor is a strided/transposed view, and safetensors
    # serializes the raw buffer — saving it non-contiguous silently corrupts it (e.g. J_regressor
    # rows lose most non-zeros), which collapses the exported skeleton to the origin. Force a copy.
    return np.ascontiguousarray(x, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-betas", type=int, default=10)
    ap.add_argument("--num-face-id", type=int, default=100,
                    help="identity shapedirs exported for the face front-end (face_id_dirs)")
    args = ap.parse_args()

    m = load_model(args.inp)

    v_template = to_np(m["v_template"])                 # [V,3]
    # SMPL-X shapedirs is [V,3,~400]: the leading ~300 are IDENTITY shape (incl. the face), the
    # tail ~100 are EXPRESSION. Keeping more identity dims is what lets the mesh represent a real,
    # personalized FACE (10 betas only captures coarse body proportions). --num-betas controls it;
    # default raised so the face front-end has geometry to fit into.
    shapedirs_full = to_np(m["shapedirs"])
    nshape = min(args.num_betas, shapedirs_full.shape[2])
    shapedirs = shapedirs_full[:, :, :nshape]           # [V,3,nbetas]
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

    # --- Face front-end (the novel identity estimator, ncg-recon/face_identity) ---------------
    # SMPL-X's 400 shapedirs split 300 IDENTITY + 100 EXPRESSION. The face module needs both bases
    # (to factor a shared neutral identity out of per-photo expression) plus the 51-landmark
    # embedding (lmk_faces_idx + lmk_bary_coords) to sample those bases at the face landmarks.
    NSPLIT = 300  # SMPL-X identity/expression boundary
    if shapedirs_full.shape[2] >= NSPLIT:
        n_id = min(args.num_face_id, NSPLIT)
        tensors["face_id_dirs"] = np.ascontiguousarray(shapedirs_full[:, :, :n_id], dtype=np.float32)
        tensors["face_expr_dirs"] = np.ascontiguousarray(shapedirs_full[:, :, NSPLIT:], dtype=np.float32)
    if "lmk_faces_idx" in m and "lmk_bary_coords" in m:
        tensors["lmk_faces_idx"] = np.ascontiguousarray(np.asarray(m["lmk_faces_idx"]), dtype=np.int64)
        tensors["lmk_bary_coords"] = np.ascontiguousarray(np.asarray(m["lmk_bary_coords"]), dtype=np.float32)
    # UV layout (texture coords + texture-face indices) -> per-texel albedo + textured glTF export.
    # `vt` [n_uv,2] are UV coords; `ft` [F,3] index into `vt` (separate from geometry `f` at seams).
    if "vt" in m:
        tensors["uv_coords"] = np.ascontiguousarray(np.asarray(m["vt"]), dtype=np.float32)
    if "ft" in m:
        tensors["uv_faces"] = np.ascontiguousarray(np.asarray(m["ft"]), dtype=np.int64)
    for k, v in tensors.items():
        print(f"  {k}: {v.shape} {v.dtype}")
    save_file(tensors, args.out, metadata={"__ncg__": f"SMPL-X from {args.inp}"})
    print(f"wrote SMPL-X -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

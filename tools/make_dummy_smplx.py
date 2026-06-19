#!/usr/bin/env python3
"""Write a runnable DUMMY SMPL-X model (a UV sphere, single root joint) to safetensors.

Lets ncg_viewer / ncg_cli render + record end-to-end WITHOUT the license-gated official
SMPL-X model. Clearly not a real body — use tools/convert_smplx.py for that.

    python tools/make_dummy_smplx.py --out data/weights/smplx_dummy.safetensors
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from safetensors.numpy import save_file


def uv_sphere(n_lat: int = 24, n_lon: int = 48, radius: float = 0.9):
    lat = np.linspace(0.0, np.pi, n_lat)
    lon = np.linspace(0.0, 2.0 * np.pi, n_lon)
    verts = []
    for a in lat:
        for b in lon:
            verts.append([radius * np.sin(a) * np.cos(b),
                          radius * np.cos(a),            # y up
                          radius * np.sin(a) * np.sin(b)])
    return np.asarray(verts, dtype=np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--betas", type=int, default=10)
    args = ap.parse_args()

    v = uv_sphere()
    V = v.shape[0]
    J = 1                 # single root joint
    nb = args.betas

    tensors = {
        "v_template": v,                                            # [V,3]
        "shapedirs": np.zeros((V, 3, nb), dtype=np.float32),        # [V,3,nbetas]
        "posedirs": np.zeros((V, 3, 9 * (J - 1)), dtype=np.float32),  # [V,3,0]
        "J_regressor": np.full((J, V), 1.0 / V, dtype=np.float32),  # mean -> body center
        "lbs_weights": np.ones((V, J), dtype=np.float32),           # all to root
        "parents": np.zeros((J,), dtype=np.int64),                  # root parent ignored
    }
    save_file(tensors, args.out, metadata={"__ncg__": "dummy SMPL-X (UV sphere)"})
    print(f"wrote dummy SMPL-X: V={V} J={J} betas={nb} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

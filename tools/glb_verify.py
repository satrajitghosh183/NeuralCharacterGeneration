#!/usr/bin/env python3
"""Game-ready glb verification (TASK 1): skeleton integrity, dressable skin weights, material opacity.
Structural/numeric gates (cheap, no ML). Render-based gates (body look, animation) need a viewer."""
import sys

import numpy as np
from pygltflib import GLTF2

CTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def accessor_array(g, blob, idx):
    a = g.accessors[idx]
    bv = g.bufferViews[a.bufferView]
    off = (bv.byteOffset or 0) + (a.byteOffset or 0)
    n = a.count * NCOMP[a.type]
    arr = np.frombuffer(blob, dtype=CTYPE[a.componentType], count=n, offset=off)
    return arr.reshape(a.count, NCOMP[a.type])


def main():
    g = GLTF2().load(sys.argv[1])
    blob = g.binary_blob()
    print(f"== {sys.argv[1]} ==")
    print(f"meshes={len(g.meshes)} nodes={len(g.nodes)} skins={len(g.skins)} anims={len(g.animations)}")

    # GATE 1 — skeleton
    ok1 = False
    if g.skins:
        joints = g.skins[0].joints
        child = set()
        for n in g.nodes:
            for c in (n.children or []):
                child.add(c)
        roots = [i for i in range(len(g.nodes)) if i not in child]
        named = sum(1 for j in joints if (g.nodes[j].name or "").strip())
        print(f"GATE1 skeleton: joints={len(joints)} named={named}/{len(joints)} scene_roots={len(roots)}")
        ok1 = len(joints) >= 50 and named == len(joints)
    else:
        print("GATE1 skeleton: NO SKIN")

    # GATE 2 — skin weights (dressable)
    ok2 = False
    prim = g.meshes[0].primitives[0]
    attrs = prim.attributes
    if getattr(attrs, "WEIGHTS_0", None) is not None:
        W = accessor_array(g, blob, attrs.WEIGHTS_0).astype(np.float32)
        J = accessor_array(g, blob, attrs.JOINTS_0)
        # if weights are normalized ubyte, scale
        if W.dtype != np.float32 or W.max() > 1.5:
            W = W / 255.0
        sums = W.sum(1)
        nan = int(np.isnan(W).sum())
        infl = int((W > 1e-4).sum(1).max())
        print(f"GATE2 weights: verts={len(W)} sum[min={sums.min():.3f} max={sums.max():.3f}] "
              f"max_influences={infl} NaN={nan}")
        ok2 = abs(sums.mean() - 1.0) < 0.05 and infl <= 4 and nan == 0
    else:
        print("GATE2 weights: NO WEIGHTS_0")

    # GATE 3 — material opacity (opaque/neutral vs translucent)
    ok3 = False
    if g.materials:
        m = g.materials[0]
        am = getattr(m, "alphaMode", None) or "OPAQUE"
        pbr = m.pbrMetallicRoughness
        bc = getattr(pbr, "baseColorFactor", None) if pbr else None
        print(f"GATE3 material: alphaMode={am} baseColorFactor={bc} doubleSided={getattr(m,'doubleSided',None)}")
        ok3 = am == "OPAQUE"
    else:
        print("GATE3 material: NO MATERIAL")

    # vertex / triangle counts
    posA = g.accessors[prim.attributes.POSITION]
    print(f"geom: vertices={posA.count} (POSITION)")

    print(f"\nSUMMARY: GATE1_skeleton={'PASS' if ok1 else 'CHECK'} "
          f"GATE2_dressable_weights={'PASS' if ok2 else 'CHECK'} "
          f"GATE3_opaque={'PASS' if ok3 else 'CHECK'}")


if __name__ == "__main__":
    main()

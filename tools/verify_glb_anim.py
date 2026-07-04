#!/usr/bin/env python3
"""ANIMATED rig verification (dev-side tooling, like glb_verify.py): pose a joint and check the
skinned result. The eyes/hair explosion of 2026-06-29 passed every bind-pose check and failed the
moment a bone moved — this is the gate that was missing. Rotates the head joint by 30 deg, applies
LBS exactly as an engine would (world = FK(node transforms), skin = sum w_i * W_i * IBM_i * v), and
verifies: (1) head-bound verts (incl. appended eyeballs/hair) stay within a sane radius of the head
joint; (2) body verts far from the head DON'T move; (3) no vert lands > max_r from its bind pose.

  python tools/verify_glb_anim.py <file.glb> [--joint 15] [--deg 30] [--tail N]
      --tail N: additionally report the last N verts (the appended eyes/hair geometry) separately.
"""
import argparse
import sys

import numpy as np
from pygltflib import GLTF2

CT = {5121: np.uint8, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NC = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def acc(g, blob, i):
    a = g.accessors[i]
    bv = g.bufferViews[a.bufferView]
    off = (bv.byteOffset or 0) + (a.byteOffset or 0)
    return np.frombuffer(blob, CT[a.componentType], a.count * NC[a.type], off).reshape(a.count, NC[a.type])


def rot_x(deg):
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    m = np.eye(4)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("glb")
    ap.add_argument("--joint", type=int, default=15, help="skin-joint slot to rotate (15=head)")
    ap.add_argument("--deg", type=float, default=30.0)
    ap.add_argument("--tail", type=int, default=0, help="report the last N verts separately")
    args = ap.parse_args()

    g = GLTF2().load(args.glb)
    blob = g.binary_blob()
    p = g.meshes[0].primitives[0]
    V = acc(g, blob, p.attributes.POSITION).astype(np.float64)
    J = acc(g, blob, p.attributes.JOINTS_0).astype(np.int64)
    W = acc(g, blob, p.attributes.WEIGHTS_0).astype(np.float64)
    if W.max() > 1.5:
        W = W / 255.0
    skin = g.skins[0]
    joints = skin.joints
    ibm = acc(g, blob, skin.inverseBindMatrices).astype(np.float64).reshape(-1, 4, 4).transpose(0, 2, 1)

    # FK over node hierarchy with the test rotation applied at the chosen joint's node.
    N = len(g.nodes)
    local = [np.eye(4) for _ in range(N)]
    for i, n in enumerate(g.nodes):
        if n.matrix:
            local[i] = np.array(n.matrix).reshape(4, 4).T
        elif n.translation:
            local[i][:3, 3] = n.translation
    target_node = joints[args.joint]
    local[target_node] = local[target_node] @ rot_x(args.deg)
    parent = [-1] * N
    for i, n in enumerate(g.nodes):
        for c in n.children or []:
            parent[c] = i
    world = [None] * N

    def fk(i):
        if world[i] is None:
            world[i] = local[i] if parent[i] == -1 else fk(parent[i]) @ local[i]
        return world[i]

    mats = np.stack([fk(j) @ ibm[k] for k, j in enumerate(joints)])  # [J,4,4] skinning matrices

    Vh = np.concatenate([V, np.ones((len(V), 1))], 1)  # [n,4]
    skinned = np.zeros_like(V)
    for c in range(J.shape[1]):
        M = mats[J[:, c]]  # [n,4,4]
        skinned += W[:, c:c + 1] * np.einsum("nij,nj->ni", M, Vh)[:, :3]

    disp = np.linalg.norm(skinned - V, axis=1)
    # head-joint rest position (translation part of the inverse of its IBM)
    head_rest = np.linalg.inv(ibm[args.joint])[:3, 3]
    head_now = fk(target_node)[:3, 3]

    hb = W[np.arange(len(V)), :][:, :] * (J == args.joint)  # weight on the rotated joint per vert
    headw = (W * (J == args.joint)).sum(1)
    head_verts = headw > 0.5
    far_body = (np.linalg.norm(V - head_rest, axis=1) > 0.6) & (headw < 1e-4)

    print(f"== {args.glb}  rotate joint {args.joint} by {args.deg} deg ==")
    print(f"verts={len(V)}  head-bound={int(head_verts.sum())}  far-body={int(far_body.sum())}")
    print(f"GATE1 head-bound verts stay near head: max dist from head joint after pose = "
          f"{np.linalg.norm(skinned[head_verts] - head_now, axis=1).max():.3f} m (must be < 0.40)")
    print(f"GATE2 far body unmoved: max displacement = {disp[far_body].max():.5f} m (must be < 0.001)")
    print(f"GATE3 global sanity: max displacement anywhere = {disp.max():.3f} m (must be < 0.60)")
    ok = (np.linalg.norm(skinned[head_verts] - head_now, axis=1).max() < 0.40 and
          disp[far_body].max() < 1e-3 and disp.max() < 0.60)
    if args.tail > 0:
        t = slice(len(V) - args.tail, len(V))
        td = np.linalg.norm(skinned[t] - head_now, axis=1)
        print(f"TAIL ({args.tail} appended verts, eyes/hair): dist-from-head after pose "
              f"min={td.min():.3f} max={td.max():.3f} (must be < 0.40) | "
              f"moved with head: {(np.linalg.norm(skinned[t] - V[t], axis=1) > 1e-4).mean() * 100:.0f}%")
        ok = ok and td.max() < 0.40
    print("ANIMATED-GATE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

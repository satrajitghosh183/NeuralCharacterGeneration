#!/usr/bin/env python3
"""Forensic diagnostic of the avatar artifacts. Emits metrics.jsonl. No prose — numbers only."""
import argparse, hashlib, json, re, struct, sys
import numpy as np
from PIL import Image


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def parse_ply(path):
    d = open(path, "rb").read()
    he = d.index(b"end_header\n") + len(b"end_header\n")
    hdr = d[:he].decode("latin1")
    N = int(re.search(r"element vertex (\d+)", hdr).group(1))
    props = re.findall(r"property float (\w+)", hdr)
    body = np.frombuffer(d[he:he + N * len(props) * 4], dtype=np.float32).reshape(N, len(props))
    return N, props, body


def parse_glb(path):
    d = open(path, "rb").read()
    ln = struct.unpack("<I", d[12:16])[0]
    return json.loads(d[20:20 + ln])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--ply2", default="")  # other ply for byte-identical check
    ap.add_argument("--faces", required=True)  # faces npy (topology)
    ap.add_argument("--out", default="/tmp/diag.json")
    a = ap.parse_args()
    R, P = a.dir, a.prefix
    M = {}

    # ---------- D1: PLY provenance ----------
    N, props, body = parse_ply(f"{R}/{P}_character.ply")
    M["D1_ply_N"] = N
    M["D1_is_smplx_vertcount"] = (N == 10475)
    oi = props.index("opacity"); op = body[:, oi]
    M["D1_opacity_mean"] = round(float(op.mean()), 4)
    M["D1_opacity_std"] = round(float(op.std()), 6)
    M["D1_opacity_unique"] = int(np.unique(np.round(op, 4)).size)
    si = [props.index(f"scale_{k}") for k in range(3)]; sc = body[:, si]
    M["D1_scale_min"] = round(float(sc.min()), 5); M["D1_scale_max"] = round(float(sc.max()), 5)
    xyz = body[:, :3]
    M["D1_bbox_height_u"] = round(float(xyz[:, 1].max() - xyz[:, 1].min()), 4)
    M["D1_md5"] = md5(f"{R}/{P}_character.ply")
    if a.ply2:
        M["D1_md5_other"] = md5(a.ply2)
        M["D1_ply_byte_identical"] = (M["D1_md5"] == M["D1_md5_other"])

    # Δv — THE KEY TEST
    dv = np.load(f"{R}/{P}_delta_v.npy")
    dvn = np.linalg.norm(dv, axis=1)
    M["D1_dv_max_u"] = round(float(dvn.max()), 6)
    M["D1_dv_mean_u"] = round(float(dvn.mean()), 6)
    M["D1_dv_p99_u"] = round(float(np.percentile(dvn, 99)), 6)
    M["D1_dv_nonzero_frac"] = round(float((dvn > 1e-5).mean()), 4)
    M["D1_contribution1_geometry_present"] = bool(dvn.max() > 1e-4)

    # ---------- D2: geometry resolution ----------
    verts = np.load(f"{R}/{P}_verts.npy")
    faces = np.load(a.faces).astype(np.int64)
    y = verts[:, 1]
    total_h = float(y.max() - y.min())
    head_mask = y > np.quantile(y, 0.86)  # head ~ top 14% of body height
    M["D2_head_vert_count"] = int(head_mask.sum())
    fm = head_mask[faces].all(1)
    M["D2_head_face_count"] = int(fm.sum())
    fv = verts[faces[fm]]
    e = np.concatenate([fv[:, 0] - fv[:, 1], fv[:, 1] - fv[:, 2], fv[:, 2] - fv[:, 0]])
    el = np.linalg.norm(e, axis=1)
    # scale: a real head is ~22cm tall; head band spans (y.max - quantile) units
    head_h_u = float(y.max() - np.quantile(y, 0.86))
    mm_per_u = 220.0 / max(head_h_u, 1e-6)
    M["D2_total_height_u"] = round(total_h, 4)
    M["D2_mean_head_edge_mm"] = round(float(el.mean()) * mm_per_u, 2)
    M["D2_median_head_edge_mm"] = round(float(np.median(el)) * mm_per_u, 2)

    # ---------- D3/D4: UV texture ----------
    tex = np.asarray(Image.open(f"{R}/{P}_albedo_uv.png").convert("RGB"), np.float32) / 255.0
    lum = tex.mean(2)
    valid = lum > 0.04
    M["D4_uv_valid_frac"] = round(float(valid.mean()), 4)
    M["D4_uv_black_frac"] = round(float((lum <= 0.02).mean()), 4)
    mx = tex.max(2); mn = tex.min(2)
    chroma = (mx - mn) / np.clip(mx, 1e-3, None)
    skin = valid & (lum > 0.25) & (lum < 0.85)
    M["D3_chroma_mean_skin"] = round(float(chroma[skin].mean()), 4)
    M["D3_chroma_p90_skin"] = round(float(np.percentile(chroma[skin], 90)), 4)
    M["D3_chroma_highfrac"] = round(float((chroma[skin] > 0.35).mean()), 4)  # specular-baked frac
    # adjacency color delta (faceting / seam jumps), on valid neighbors
    dx = np.abs(tex[:, 1:] - tex[:, :-1]).sum(2); vmx = valid[:, 1:] & valid[:, :-1]
    dy = np.abs(tex[1:] - tex[:-1]).sum(2); vmy = valid[1:] & valid[:-1]
    adj = np.concatenate([dx[vmx], dy[vmy]])
    M["D4_adj_delta_mean"] = round(float(adj.mean()), 4)
    M["D4_adj_delta_p95"] = round(float(np.percentile(adj, 95)), 4)
    M["D4_adj_seam_frac"] = round(float((adj > 0.30).mean()), 4)  # hard cross-edge jumps

    # ---------- D6: glb sanity ----------
    try:
        g = parse_glb(f"{R}/{P}_char_textured.glb")
        M["D6_glb_joints"] = len(g["skins"][0]["joints"]) if g.get("skins") else 0
        M["D6_glb_animations"] = len(g.get("animations", []))
        M["D6_glb_has_normaltex"] = any("normalTexture" in m for m in g.get("materials", []))
        M["D6_glb_has_uv"] = any("TEXCOORD_0" in pr.get("attributes", {})
                                 for me in g.get("meshes", []) for pr in me["primitives"])
    except Exception as e:
        M["D6_glb_error"] = str(e)

    json.dump(M, open(a.out, "w"), indent=2)
    for k, v in M.items():
        print(f"{k} = {v}")


if __name__ == "__main__":
    main()

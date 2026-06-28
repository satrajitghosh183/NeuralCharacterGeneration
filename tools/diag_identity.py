#!/usr/bin/env python3
"""D5 identity: ArcFace cosine of an avatar render vs the subject's real photos, vs real-vs-real."""
import argparse, glob, numpy as np, torch, torchvision  # noqa
from PIL import Image
import torch.nn.functional as F

TEMPLATE = np.array([[38.0, 52.0], [74.0, 52.0], [56.0, 72.0], [56.0, 92.0]], np.float32)  # eyes,nose,mouth


def umeyama(src, dst):
    sm, dm = src.mean(0), dst.mean(0)
    sc, dc = src - sm, dst - dm
    H = dc.T @ sc / len(src)
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = U @ Vt
    s = S.sum() / (sc * sc).sum() * len(src)
    t = dm - s * R @ sm
    M = np.eye(3); M[:2, :2] = s * R; M[:2, 2] = t
    return M


def embed(img01, det, arc, dev):  # img01 [3,H,W] cuda
    H, W = img01.shape[1], img01.shape[2]
    with torch.no_grad():
        boxes = det(img01)
    if boxes.shape[0] == 0:
        return None
    b = boxes[0].cpu().numpy()
    kp = b[4:16].reshape(6, 2)  # blazeface: r_eye,l_eye,nose,mouth,r_ear,l_ear
    src = np.stack([kp[0], kp[1], kp[2], kp[3]]).astype(np.float32)
    M = umeyama(src, TEMPLATE)
    # inverse-warp the 112x112 aligned crop via grid_sample
    ys, xs = np.meshgrid(np.arange(112), np.arange(112), indexing="ij")
    pts = np.stack([xs.ravel(), ys.ravel(), np.ones(112 * 112)], 0)
    inv = np.linalg.inv(M) @ pts
    gx = inv[0].reshape(112, 112) / (W - 1) * 2 - 1
    gy = inv[1].reshape(112, 112) / (H - 1) * 2 - 1
    grid = torch.tensor(np.stack([gx, gy], -1)[None], dtype=torch.float32, device=dev)
    crop = F.grid_sample(img01[None], grid, align_corners=True)  # [1,3,112,112]
    with torch.no_grad():
        e = arc((crop * 2 - 1)).reshape(-1)
    return (e / e.norm().clamp_min(1e-9)).cpu().numpy()


def load01(p, dev):
    return torch.tensor(np.asarray(Image.open(p).convert("RGB"), np.float32).transpose(2, 0, 1) / 255,
                        device=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avatar", nargs="+", required=True)
    ap.add_argument("--album", required=True)
    ap.add_argument("--detector", required=True); ap.add_argument("--arcface", required=True)
    ap.add_argument("--n-real", type=int, default=12)
    a = ap.parse_args()
    dev = "cpu"  # traced detector has CPU weights baked in
    det = torch.jit.load(a.detector).eval()
    arc = torch.jit.load(a.arcface).eval()

    reals = []
    for p in sorted(glob.glob(f"{a.album}/*.jpg"))[: a.n_real * 3]:
        try:
            e = embed(load01(p, dev), det, arc, dev)
        except Exception:
            e = None
        if e is not None:
            reals.append(e)
        if len(reals) >= a.n_real:
            break
    reals = np.array(reals)
    # real-vs-real mean cosine (off-diagonal)
    C = reals @ reals.T
    rr = (C.sum() - np.trace(C)) / (len(reals) * (len(reals) - 1))
    print(f"D5_real_count = {len(reals)}")
    print(f"D5_real_vs_real_cos = {rr:.4f}")
    rmean = reals.mean(0)
    rmean = rmean / np.linalg.norm(rmean)
    for av in a.avatar:
        try:
            ea = embed(load01(av, dev), det, arc, dev)
            cos = float(ea @ rmean) if ea is not None else float("nan")
            print(f"D5_avatar_vs_real_cos[{av.split('/')[-1]}] = {cos:.4f}")
        except Exception as ex:
            print(f"D5_avatar[{av}] = FAIL {ex}")


if __name__ == "__main__":
    main()

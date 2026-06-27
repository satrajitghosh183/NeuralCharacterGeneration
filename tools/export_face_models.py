#!/usr/bin/env python3
"""Export the Phase-A face front-end nets to TorchScript matching the FIXED C++ I/O contract
(ncg/body/face_models.hpp). DEV-ONLY (no Python at runtime). Verified working with:
  detector : BlazeFace      — github.com/hollance/BlazeFace-PyTorch  (blazeface.pth + anchors.npy)
  facemesh : FaceMesh 468   — github.com/thepowerfuldeez/facemesh.pytorch  (facemesh.pth)
  embedder : InceptionResnetV1 vggface2 — pip facenet-pytorch (identity embedding for the gate)

C++ contract (must match exactly):
  detector(img[3,H,W] f32 in [0,1])      -> boxes [N,5] = (x0,y0,x1,y1,score), source-image px
  facemesh(crop[1,3,192,192] f32 in[0,1])-> (uv[468,2] in [0,1] crop-space, conf[468] in [0,1])
  arcface (crop[1,3,112,112] f32 in[0,1])-> emb[512] (L2-normalized)

Usage (on the H100, in the face_venv with the two repos cloned):
  python tools/export_face_models.py --face-src <dir with facemesh.pytorch + BlazeFace-PyTorch> \
                                     --out data/face
Outputs: <out>/{detector,facemesh,arcface}.torchscript  (loaded by ncg::body::*::load).
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


class MeshW(nn.Module):
    """FaceMesh: [1,3,192,192] in [0,1] -> (uv[468,2] in [0,1], conf[468])."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        r, c = self.net(x * 2 - 1)                       # r[1,1404], c[1,1]
        uv = r[0].reshape(468, 3)[:, :2] / 192.0
        conf = torch.sigmoid(c[0]).reshape(1).expand(468).contiguous()
        return uv.contiguous(), conf


class EmbW(nn.Module):
    """ArcFace identity embedder: ALIGNED [1,3,112,112] in [0,1] -> emb[512] (L2). ArcFace is 112²
    native (no resize); expects (x-0.5)/0.5 normalization."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        e = self.net(x * 2 - 1).reshape(-1)
        return e / e.norm().clamp_min(1e-9)


class Det(nn.Module):
    """BlazeFace detector: [3,H,W] in [0,1] -> [N, 5+12] = (x0,y0,x1,y1,score, kp0x,kp0y..kp5x,kp5y)
    in source px. The 6 keypoints (right eye, left eye, nose, mouth, R-ear, L-ear) drive face
    ALIGNMENT before the identity embedder. Letterbox 128, decode anchors, threshold, NMS. Scripted."""
    def __init__(self, net, anchors, thr: float = 0.5, iou: float = 0.3):
        super().__init__()
        self.net = net
        self.register_buffer("anchors", anchors)
        self.thr = thr
        self.iou = iou

    def forward(self, img):
        H = img.size(1)
        W = img.size(2)
        m = float(H if H > W else W)
        sc = 128.0 / m
        nh = int(float(H) * sc)
        nw = int(float(W) * sc)
        x = F.interpolate(img.unsqueeze(0), size=[nh, nw], mode="bilinear", align_corners=False)
        canvas = torch.zeros([1, 3, 128, 128])
        canvas[:, :, :nh, :nw] = x
        out = self.net(canvas * 2 - 1)
        r = out[0]
        c = out[1]                                       # r[1,896,16], c[1,896,1]
        a = self.anchors
        xc = r[0, :, 0] / 128.0 * a[:, 2] + a[:, 0]
        yc = r[0, :, 1] / 128.0 * a[:, 3] + a[:, 1]
        w = r[0, :, 2] / 128.0 * a[:, 2]
        h = r[0, :, 3] / 128.0 * a[:, 3]
        x0 = (xc - w / 2) * m
        y0 = (yc - h / 2) * m
        x1 = (xc + w / 2) * m
        y1 = (yc + h / 2) * m
        score = c[0, :, 0].clamp(-100.0, 100.0).sigmoid()
        cols = [x0, y0, x1, y1, score]
        for k in range(6):                                # 6 keypoints, decoded to source px
            kx = (r[0, :, 4 + 2 * k] / 128.0 * a[:, 2] + a[:, 0]) * m
            ky = (r[0, :, 5 + 2 * k] / 128.0 * a[:, 3] + a[:, 1]) * m
            cols.append(kx)
            cols.append(ky)
        full = torch.stack(cols, dim=1)                   # [896, 17]
        keep = score >= self.thr
        full = full[keep]
        if full.size(0) == 0:
            return torch.zeros([0, 17])
        idx = nms(full[:, 0:4], full[:, 4], self.iou)
        return full[idx]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--face-src", required=True, help="dir containing facemesh.pytorch/ and BlazeFace-PyTorch/")
    ap.add_argument("--out", default="data/face")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fmdir = os.path.join(args.face_src, "facemesh.pytorch")
    bfdir = os.path.join(args.face_src, "BlazeFace-PyTorch")
    sys.path.insert(0, fmdir)
    sys.path.insert(0, bfdir)

    from facemesh import FaceMesh
    fm = FaceMesh(); fm.load_weights(os.path.join(fmdir, "facemesh.pth")); fm.eval()
    torch.jit.trace(MeshW(fm), torch.rand(1, 3, 192, 192), check_trace=False).save(
        os.path.join(args.out, "facemesh.torchscript")); print("facemesh OK")

    # ArcFace IR-SE50 (facexlib weights) — far more discriminative than facenet/vggface2 on casual
    # crops. Loaded by direct file import to bypass facexlib's cv2-dependent package __init__.
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "arcface_arch", os.path.join(args.face_src, "facexlib/facexlib/recognition/arcface_arch.py"))
    _arc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_arc)
    emb = _arc.Backbone(num_layers=50, drop_ratio=0.6, mode="ir_se")
    emb.load_state_dict(torch.load(os.path.join(args.face_src, "arcface_ir_se50.pth"), map_location="cpu"))
    emb.eval()
    torch.jit.trace(EmbW(emb), torch.rand(1, 3, 112, 112), check_trace=False).save(
        os.path.join(args.out, "arcface.torchscript")); print("arcface OK")

    from blazeface import BlazeFace
    bf = BlazeFace(); bf.load_weights(os.path.join(bfdir, "blazeface.pth"))
    bf.load_anchors(os.path.join(bfdir, "anchors.npy")); bf.eval()
    traced = torch.jit.trace(bf, torch.rand(1, 3, 128, 128) * 2 - 1, check_trace=False)
    anchors = torch.from_numpy(np.load(os.path.join(bfdir, "anchors.npy"))).float()
    torch.jit.script(Det(traced, anchors)).save(os.path.join(args.out, "detector.torchscript"))
    print("detector OK")
    print("wrote", args.out, "/{detector,facemesh,arcface}.torchscript")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

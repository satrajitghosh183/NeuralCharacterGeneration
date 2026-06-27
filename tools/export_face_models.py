#!/usr/bin/env python3
"""Export the Phase-A face front-end nets to TorchScript matching the FIXED C++ I/O contract
(ncg/body/face_models.hpp). DEV-ONLY (no Python at runtime). Each model is wrapped so the C++
stays architecture-agnostic; only this tool knows the underlying nets.

C++ contract (must match exactly):
  detector(img[3,H,W] f32 in [0,1])      -> boxes [N,5] = (x0,y0,x1,y1,score), source-image px
  facemesh(crop[3,192,192] f32 in [0,1]) -> (uv[K,2] in [0,1] crop-space, conf[K] in [0,1])
  arcface (crop[3,112,112] f32 in [0,1]) -> emb[512] (L2-normalized)

Recommended torch-native sources (all export to TorchScript; pick what you have on the box):
  detector : RetinaFace  — github.com/biubug6/Pytorch_Retinaface (single scriptable net)
  facemesh : FaceMesh468 — github.com/thepowerfuldeez/facemesh.pytorch (MediaPipe weights, 468 pts)
  arcface  : ArcFace r100 — insightface arcface_torch, OR facenet-pytorch InceptionResnetV1 (pip)

Usage (on the H100, dev env with torch + the chosen model repos/weights):
  python tools/export_face_models.py --out data/face \
     --retinaface-weights <...>.pth --facemesh-weights facemesh.pth --arcface-weights <...>.pth
Outputs: data/face/{detector,facemesh,arcface}.torchscript  (loaded by ncg::body::*::load).
Each exporter is isolated: a missing model just skips that one (the C++ test SKIPs on missing asset).
"""
import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- detector (RetinaFace)
class DetectorWrap(nn.Module):
    """Wrap a RetinaFace net to the contract: full image -> [N,5] (x0,y0,x1,y1,score) in src px.
    Internally letterboxes to the net's input, runs, decodes anchors, rescales boxes to src px."""
    def __init__(self, net, in_size=640, score_thr=0.5):
        super().__init__()
        self.net = net
        self.in_size = in_size
        self.score_thr = score_thr

    def forward(self, img):  # img [3,H,W] in [0,1]
        _, H, W = img.shape
        s = self.in_size / max(H, W)
        nh, nw = int(round(H * s)), int(round(W * s))
        x = F.interpolate(img.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False)
        canvas = torch.zeros(1, 3, self.in_size, self.in_size, dtype=img.dtype, device=img.device)
        canvas[:, :, :nh, :nw] = x
        # NOTE: RetinaFace returns (loc, conf, landms); decoding with priors is model-specific.
        # Replace the next two lines with the repo's decode() to get [N,5] in canvas px, then /s.
        boxes_canvas = self.net(canvas)  # expected [N,5] (x0,y0,x1,y1,score) after the repo's decode
        boxes = boxes_canvas.clone()
        boxes[:, :4] = boxes[:, :4] / s
        return boxes[boxes[:, 4] >= self.score_thr]


# ----------------------------------------------------------------------------- facemesh (468 pts)
class FaceMeshWrap(nn.Module):
    """Wrap MediaPipe-FaceMesh-pytorch to the contract. The net takes a 192x192 crop and returns
    468x3 landmarks (x,y in [0,192], z) + a face-presence flag. We emit uv in [0,1] and broadcast
    the presence score as per-point confidence (MediaPipe gives one flag, not per-point)."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, crop):  # crop [3,192,192] in [0,1]
        out = self.net(crop.unsqueeze(0) * 255.0 if crop.max() <= 1.0 else crop.unsqueeze(0))
        lmk, flag = out  # lmk [1,468,3] in [0,192], flag [1,1]
        uv = lmk[0, :, :2] / 192.0                          # [K,2] in [0,1]
        conf = torch.sigmoid(flag).reshape(1).expand(uv.shape[0]).contiguous()  # [K]
        return (uv.contiguous(), conf)


# ----------------------------------------------------------------------------- arcface (512-d emb)
class ArcFaceWrap(nn.Module):
    """Wrap an ArcFace/embedding backbone to the contract. Applies the standard (x-0.5)/0.5
    normalization a 112x112 face expects, returns an L2-normalized 512-d embedding."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, crop):  # crop [3,112,112] in [0,1]
        x = (crop.unsqueeze(0) - 0.5) / 0.5
        e = self.net(x).reshape(-1)
        return e / e.norm().clamp_min(1e-9)


def _save(mod, example, path):
    mod.eval()
    with torch.no_grad():
        ts = torch.jit.trace(mod, example, check_trace=False)
    torch.jit.save(ts, path)
    print(f"  wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/face")
    ap.add_argument("--retinaface-weights")
    ap.add_argument("--facemesh-weights")
    ap.add_argument("--arcface-weights")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Each block is best-effort and isolated: it loads the underlying net from the user-provided
    # weights, wraps to the contract, and traces. Fill in the repo-specific net construction.
    if args.facemesh_weights:
        try:
            from facemesh import FaceMesh  # thepowerfuldeez/facemesh.pytorch on PYTHONPATH
            net = FaceMesh(); net.load_weights(args.facemesh_weights)
            _save(FaceMeshWrap(net), torch.rand(3, 192, 192), os.path.join(args.out, "facemesh.torchscript"))
        except Exception as e:
            print(f"  [facemesh] skipped: {e}", file=sys.stderr)

    if args.arcface_weights:
        try:
            # e.g. insightface arcface_torch: from backbones import get_model; net=get_model('r100')
            net = torch.load(args.arcface_weights, map_location="cpu")
            net = net.eval() if isinstance(net, nn.Module) else net
            _save(ArcFaceWrap(net), torch.rand(3, 112, 112), os.path.join(args.out, "arcface.torchscript"))
        except Exception as e:
            print(f"  [arcface] skipped: {e}", file=sys.stderr)

    if args.retinaface_weights:
        try:
            print("  [detector] construct RetinaFace + decode() per the repo, then wrap.", file=sys.stderr)
            # net = load_retinaface(args.retinaface_weights)  # repo-specific
            # _save(DetectorWrap(net), torch.rand(3, 640, 640), os.path.join(args.out, "detector.torchscript"))
        except Exception as e:
            print(f"  [detector] skipped: {e}", file=sys.stderr)

    print("done. Point ncg_cli at data/face/{detector,facemesh,arcface}.torchscript")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

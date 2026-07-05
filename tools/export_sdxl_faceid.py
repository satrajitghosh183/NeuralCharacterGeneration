#!/usr/bin/env python3
"""Export RealVisXL + IP-Adapter-FaceID (identity-locked SDXL) for the C++ face refine.

DEV-side (like export_sd.py). Produces into --out-dir:
  sdxl_unet_ip.ts      forward(latent[1,4,h,w], t[1] long, ctx[1,77,2048], pooled[1,1280],
                               ip[1,K,512]) -> eps   (FaceID tokens enter via the IP processors)
  sd_vae.ts            encode/decode bundle ([0,1] convention, fp32)
  sd_cond.safetensors  {cond, uncond}[1,77,2048] + {cond_pooled, uncond_pooled}[1,1280]
                       + {ip_cond}[1,K,512] (the subject's arcface embeds from the album)
                       + {ip_uncond} zeros

The subject embeds come from the album via facexlib retinaface + the arcface ir-se50 weights
(same recognition family FaceID was trained on).

  python tools/export_sdxl_faceid.py --album data/me_clean_album --face-src face_src \
      --prompt "..." --out-dir models/sdxl_faceid_cpu --res 512
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors.torch import save_file


class UNetIPWrap(nn.Module):
    def __init__(self, unet, time_ids):
        super().__init__()
        self.unet = unet
        self.register_buffer("time_ids", time_ids)

    def forward(self, latent, t, ctx, pooled, ip):
        return self.unet(latent, t, encoder_hidden_states=ctx,
                         added_cond_kwargs={"text_embeds": pooled,
                                            "time_ids": self.time_ids,
                                            "image_embeds": [ip]}).sample


class VaeEncode(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):
        return self.vae.encode(2.0 * x - 1.0).latent_dist.mean


class VaeDecode(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return (self.vae.decode(z).sample / 2.0 + 0.5).clamp(0.0, 1.0)


class VaeBundle(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    @torch.jit.export
    def encode(self, x):
        return self.enc(x)

    @torch.jit.export
    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        return self.enc(x)


@torch.no_grad()
def album_embeds(album, face_src, top_k=6):
    """Subject arcface embeds [1,K,512] from the sharpest album faces (norm-crop 112)."""
    from facexlib.detection import init_detection_model
    from facexlib.utils.face_restoration_helper import get_largest_face
    det = init_detection_model("retinaface_resnet50", half=False, device="cpu")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "arcface_arch", os.path.join(face_src, "facexlib/facexlib/recognition/arcface_arch.py"))
    aa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aa)
    net = aa.Backbone(50, 0.6, "ir_se").eval()
    net.load_state_dict(torch.load(os.path.join(face_src, "arcface_ir_se50.pth"),
                                   map_location="cpu"), strict=True)
    embs = []
    scored = []
    for f in sorted(glob.glob(os.path.join(album, "*.jpg"))):
        im = Image.open(f).convert("RGB")
        im.thumbnail((1200, 1200))
        arr = np.asarray(im)[:, :, ::-1].copy()
        bb = det.detect_faces(arr, 0.9)
        if len(bb) == 0:
            continue
        b = max(bb, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
        scored.append((float(b[4]) * (b[2] - b[0]) * (b[3] - b[1]), f, b))
    scored.sort(reverse=True)
    for _, f, b in scored[:top_k]:
        im = Image.open(f).convert("RGB")
        im.thumbnail((1200, 1200))
        w = (b[2] - b[0]) * 0.75
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        crop = im.crop((cx - w, cy - w, cx + w, cy + w)).resize((112, 112), Image.LANCZOS)
        x = torch.from_numpy(np.asarray(crop, np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        x = (x - 0.5) / 0.5
        e = net(x)
        embs.append(torch.nn.functional.normalize(e, dim=-1))
    print(f"[sdxl_faceid] {len(embs)} identity embeds from the album")
    return torch.cat(embs, 0).unsqueeze(0)  # [1,K,512]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="SG161222/RealVisXL_V4.0")
    ap.add_argument("--album", required=True)
    ap.add_argument("--face-src", required=True)
    ap.add_argument("--prompt", default="closeup studio portrait photograph of a man's face, "
                    "detailed photorealistic skin with visible pores, natural warm skin tone, "
                    "sharp focus, soft studio lighting, high detail")
    ap.add_argument("--out-dir", default="models/sdxl_faceid_cpu")
    ap.add_argument("--res", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dt = torch.float32

    from diffusers import StableDiffusionXLImg2ImgPipeline
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(args.model, torch_dtype=dt)
    pipe.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None,
                         weight_name="ip-adapter-faceid-portrait_sdxl.bin",
                         image_encoder_folder=None)
    pipe.set_ip_adapter_scale(0.8)
    unet = pipe.unet.eval()
    vae = pipe.vae.eval()

    # prompt embeds (dual encoder)
    pe, ne, ppool, npool = pipe.encode_prompt(prompt=args.prompt, negative_prompt="",
                                              device="cpu", num_images_per_prompt=1,
                                              do_classifier_free_guidance=True)
    ip = album_embeds(args.album, args.face_src).to(dt)
    save_file({"cond": pe.contiguous(), "uncond": ne.contiguous(),
               "cond_pooled": ppool.contiguous(), "uncond_pooled": npool.contiguous(),
               "ip_cond": ip.contiguous(), "ip_uncond": torch.zeros_like(ip).contiguous()},
              os.path.join(args.out_dir, "sd_cond.safetensors"))
    print("[sdxl_faceid] cond/pooled/ip embeds saved")

    # trace the UNet with IP processors live
    h = args.res // 8
    time_ids = torch.tensor([[args.res, args.res, 0, 0, args.res, args.res]], dtype=dt)
    ex = (torch.randn(1, 4, h, h, dtype=dt), torch.tensor([500], dtype=torch.long),
          pe.to(dt), ppool.to(dt), ip)
    wrap = UNetIPWrap(unet, time_ids)
    ts = torch.jit.trace(wrap, ex, check_trace=False)
    ts.save(os.path.join(args.out_dir, "sdxl_unet_ip.ts"))
    print("[sdxl_faceid] traced UNet+FaceID -> sdxl_unet_ip.ts")

    enc = torch.jit.trace(VaeEncode(vae), (torch.rand(1, 3, args.res, args.res, dtype=dt),),
                          check_trace=False)
    dec = torch.jit.trace(VaeDecode(vae), (torch.randn(1, 4, h, h, dtype=dt),), check_trace=False)
    torch.jit.script(VaeBundle(enc, dec)).save(os.path.join(args.out_dir, "sd_vae.ts"))
    print(f"[sdxl_faceid] done -> {args.out_dir}")


if __name__ == "__main__":
    main()

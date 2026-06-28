#!/usr/bin/env python3
"""Export SD-1.5 + a NORMAL ControlNet as the TorchScript assets ncg-diffuse needs for
GEOMETRY-CONDITIONED face texturing (the real photoreal path). Plain img2img drifts/hallucinates and
seams across views; conditioning the diffusion on the rendered NORMAL MAP locks it to the face
surface, so it can add photoreal skin detail while staying consistent and on-identity. DEV-ONLY; run
with the SAME torch as the C++ LibTorch (2.6).

Produces (into --out-dir):
  control_unet.ts   forward(latent[B,4,h,w], t[B] long, ctx[B,77,C], control[B,3,H,W] in [0,1])
                    -> eps[B,4,h,w]   (ControlNet residuals folded into the UNet internally)
  sd_vae.ts         encode(rgb[0,1]) -> latent;  decode(latent) -> rgb[0,1]
  sd_cond.safetensors  { "cond":[1,77,C], "uncond":[1,77,C] }  (face-detail prompt + empty)

Usage:
  python tools/export_sd_control.py --out-dir models/sd_ctrl \
      --controlnet lllyasviel/control_v11p_sd15_normalbae \
      --prompt "a detailed photorealistic closeup photograph of a human face, skin pores, sharp focus"
"""
import argparse
import os

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, ControlNetModel, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import save_file


class ControlUNet(nn.Module):
    """ControlNet residuals folded into the UNet — one traced module: (latent,t,ctx,control)->eps."""

    def __init__(self, unet, cn):
        super().__init__()
        self.unet = unet
        self.cn = cn

    def forward(self, latent, t, ctx, control):
        down, mid = self.cn(latent, t, encoder_hidden_states=ctx, controlnet_cond=control,
                            conditioning_scale=1.0, return_dict=False)
        return self.unet(latent, t, encoder_hidden_states=ctx,
                         down_block_additional_residuals=down,
                         mid_block_additional_residual=mid).sample


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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--controlnet", default="lllyasviel/control_v11p_sd15_normalbae")
    ap.add_argument("--prompt", default="a detailed photorealistic closeup photograph of a human face, skin pores, fine skin texture, sharp focus, natural skin")
    ap.add_argument("--out-dir", default="models/sd_ctrl")
    ap.add_argument("--res", type=int, default=512)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32

    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet").to(dev, dt).eval()
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").to(dev, dt).eval()
    tok = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    txt = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder").to(dev, dt).eval()
    cn = ControlNetModel.from_pretrained(args.controlnet).to(dev, dt).eval()

    cdim = txt.config.hidden_size
    h = args.res // 8

    def embed(p):
        ids = tok(p, padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(dev)
        return txt(ids)[0].to("cpu", torch.float32)

    cond, uncond = embed(args.prompt), embed("")
    save_file({"cond": cond.contiguous(), "uncond": uncond.contiguous()},
              os.path.join(args.out_dir, "sd_cond.safetensors"))
    print(f"[export_ctrl] cond/uncond {tuple(cond.shape)} -> sd_cond.safetensors")

    ex_lat = torch.randn(1, unet.config.in_channels, h, h, device=dev, dtype=dt)
    ex_t = torch.tensor([500], device=dev, dtype=torch.long)
    ex_ctx = cond.to(dev, dt)
    ex_ctrl = torch.rand(1, 3, args.res, args.res, device=dev, dtype=dt)
    cu = ControlUNet(unet, cn)
    cu_ts = torch.jit.trace(cu, (ex_lat, ex_t, ex_ctx, ex_ctrl), check_trace=False)
    cu_ts.save(os.path.join(args.out_dir, "control_unet.ts"))
    print("[export_ctrl] traced ControlNet+UNet -> control_unet.ts")

    ex_img = torch.rand(1, 3, args.res, args.res, device=dev, dtype=dt)
    ex_z = torch.randn(1, unet.config.in_channels, h, h, device=dev, dtype=dt)
    enc = torch.jit.trace(VaeEncode(vae), (ex_img,), check_trace=False)
    dec = torch.jit.trace(VaeDecode(vae), (ex_z,), check_trace=False)
    torch.jit.script(VaeBundle(enc, dec)).save(os.path.join(args.out_dir, "sd_vae.ts"))
    print(f"[export_ctrl] VAE -> sd_vae.ts. ctx dim={cdim}. done -> {args.out_dir}/")


if __name__ == "__main__":
    main()

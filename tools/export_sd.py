#!/usr/bin/env python3
"""Export a released Stable-Diffusion checkpoint to the TorchScript + safetensors assets that
ncg-diffuse's SdGuidance loads (the SDS completion prior). DEV-ONLY tooling (workstation/H100);
NOT part of the C++ runtime. Run with the SAME torch version as the C++ LibTorch (2.6) so the
TorchScript is loadable.

Produces (into --out-dir):
  sd_unet.ts            forward(latent[B,4,h,w], t[B] long, ctx[B,77,C]) -> eps[B,4,h,w]
  sd_vae.ts             encode(rgb[B,3,H,W] in [0,1]) -> latent[B,4,h,w];  decode(latent) -> rgb[0,1]
  sd_cond.safetensors   { "cond":[1,77,C], "uncond":[1,77,C] }  (the prompt + the empty prompt)

The VAE wrapper bakes SD's [0,1]<->[-1,1] convention so the C++ side stays in [0,1]. Text embeddings
are precomputed here (CLIP tokenizer + encoder) so the C++ runtime needs no tokenizer.

Usage:
  python tools/export_sd.py --model stabilityai/stable-diffusion-2-1-base \
      --prompt "a photo of a person, full body, natural skin, neutral lighting" \
      --out-dir models/sd
"""
import argparse
import os

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from safetensors.torch import save_file


class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, latent, t, ctx):
        return self.unet(latent, t, encoder_hidden_states=ctx).sample


class VaeEncode(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, x):  # x in [0,1] -> posterior mean latent (UNscaled; C++ applies vae_scale)
        return self.vae.encode(2.0 * x - 1.0).latent_dist.mean


class VaeDecode(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):  # latent -> RGB in [0,1]
        return (self.vae.decode(z).sample / 2.0 + 0.5).clamp(0.0, 1.0)


class VaeBundle(nn.Module):
    """Scripted container exposing encode/decode (each a traced submodule) as named methods."""

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

    def forward(self, x):  # nn.Module requires a forward; encode is the sensible default
        return self.enc(x)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="stabilityai/stable-diffusion-2-1-base")
    ap.add_argument("--prompt", default="a photo of a person, full body, natural skin")
    ap.add_argument("--out-dir", default="models/sd")
    ap.add_argument("--res", type=int, default=512, help="example latent res = res/8")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float32

    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet").to(dev, dt).eval()
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae").to(dev, dt).eval()
    tok = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    txt = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder").to(dev, dt).eval()

    cdim = txt.config.hidden_size
    h = args.res // 8

    # --- text embeddings (cond + empty/uncond) ---
    def embed(p):
        ids = tok(p, padding="max_length", max_length=tok.model_max_length,
                  truncation=True, return_tensors="pt").input_ids.to(dev)
        return txt(ids)[0].to("cpu", torch.float32)

    cond, uncond = embed(args.prompt), embed("")
    save_file({"cond": cond.contiguous(), "uncond": uncond.contiguous()},
              os.path.join(args.out_dir, "sd_cond.safetensors"))
    print(f"[export_sd] cond/uncond {tuple(cond.shape)} -> sd_cond.safetensors")

    # --- UNet ---
    ex_lat = torch.randn(1, unet.config.in_channels, h, h, device=dev, dtype=dt)
    ex_t = torch.tensor([500], device=dev, dtype=torch.long)
    ex_ctx = cond.to(dev, dt)
    unet_ts = torch.jit.trace(UNetWrap(unet), (ex_lat, ex_t, ex_ctx), check_trace=False)
    unet_ts.save(os.path.join(args.out_dir, "sd_unet.ts"))
    print("[export_sd] traced UNet -> sd_unet.ts")

    # --- VAE (encode + decode bundled) ---
    ex_img = torch.rand(1, 3, args.res, args.res, device=dev, dtype=dt)
    ex_z = torch.randn(1, unet.config.in_channels, h, h, device=dev, dtype=dt)
    enc = torch.jit.trace(VaeEncode(vae), (ex_img,), check_trace=False)
    dec = torch.jit.trace(VaeDecode(vae), (ex_z,), check_trace=False)
    vae_ts = torch.jit.script(VaeBundle(enc, dec))
    vae_ts.save(os.path.join(args.out_dir, "sd_vae.ts"))
    print("[export_sd] traced VAE encode/decode -> sd_vae.ts")
    print(f"[export_sd] done. ctx dim={cdim}. Stage {args.out_dir}/ on the H100 for `ncg_cli complete`.")


if __name__ == "__main__":
    main()

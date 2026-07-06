#!/usr/bin/env python3
"""Export SD1.5 + ControlNet(normalbae) + IP-Adapter-FaceID — position-locked AND identity-locked.

The reproject bake needs the diffusion output aligned with the render (ControlNet holds the
geometry) while FaceID injects the subject's identity. Produces into --out-dir:
  control_unet_ip.ts   forward(latent[1,4,h,h], t[1], ctx[1,77,768], control[1,3,R,R],
                               ip[1,K,512]) -> eps
  sd_vae.ts            encode/decode bundle
  sd_cond.safetensors  {cond, uncond}[1,77,768] + {ip_cond}[1,K,512] + {ip_uncond} zeros

  python tools/export_sd15_faceid_control.py --album data/me_clean_album --face-src face_src \
      --out-dir models/sd15_faceid_ctrl_cpu
"""
import argparse
import os

import torch
import torch.nn as nn
from safetensors.torch import save_file

from export_sdxl_faceid import album_embeds, VaeEncode, VaeDecode, VaeBundle  # reuse


class ControlUNetIP(nn.Module):
    def __init__(self, unet, controlnet):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet

    def forward(self, latent, t, ctx, control, ip):
        down, mid = self.controlnet(latent, t, encoder_hidden_states=ctx,
                                    controlnet_cond=control, return_dict=False)
        return self.unet(latent, t, encoder_hidden_states=ctx,
                         down_block_additional_residuals=down,
                         mid_block_additional_residual=mid,
                         added_cond_kwargs={"image_embeds": [ip]}).sample


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="emilianJR/epiCRealism")
    ap.add_argument("--controlnet", default="lllyasviel/control_v11p_sd15_normalbae")
    ap.add_argument("--album", required=True)
    ap.add_argument("--face-src", required=True)
    ap.add_argument("--prompt", default="closeup studio portrait photograph of a man's face, "
                    "detailed photorealistic skin with visible pores, natural warm skin tone, "
                    "sharp focus, soft studio lighting, high detail")
    ap.add_argument("--out-dir", default="models/sd15_faceid_ctrl_cpu")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--ip-scale", type=float, default=0.9)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dt = torch.float32

    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
    cn = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dt)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.model, controlnet=cn, torch_dtype=dt, safety_checker=None,
        requires_safety_checker=False)
    pipe.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None,
                         weight_name="ip-adapter-faceid-portrait_sd15.bin",
                         image_encoder_folder=None)
    pipe.set_ip_adapter_scale(args.ip_scale)
    unet = pipe.unet.eval()
    vae = pipe.vae.eval()

    from transformers import CLIPTextModel, CLIPTokenizer
    tok = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    txt = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder").eval()

    def embed(p):
        ids = tok(p, padding="max_length", max_length=tok.model_max_length, truncation=True,
                  return_tensors="pt").input_ids
        return txt(ids)[0].float()

    ip = album_embeds(args.album, args.face_src).to(dt)
    save_file({"cond": embed(args.prompt).contiguous(), "uncond": embed("").contiguous(),
               "ip_cond": ip.contiguous(), "ip_uncond": torch.zeros_like(ip).contiguous()},
              os.path.join(args.out_dir, "sd_cond.safetensors"))
    print(f"[sd15_faceid] cond + {ip.shape[1]} identity embeds saved")

    h = args.res // 8
    ex = (torch.randn(1, 4, h, h, dtype=dt), torch.tensor([500], dtype=torch.long),
          embed(args.prompt).to(dt), torch.rand(1, 3, args.res, args.res, dtype=dt), ip)
    ts = torch.jit.trace(ControlUNetIP(unet, cn), ex, check_trace=False)
    ts.save(os.path.join(args.out_dir, "control_unet_ip.ts"))
    print("[sd15_faceid] traced ControlNet+UNet+FaceID -> control_unet_ip.ts")

    enc = torch.jit.trace(VaeEncode(vae), (torch.rand(1, 3, args.res, args.res, dtype=dt),),
                          check_trace=False)
    dec = torch.jit.trace(VaeDecode(vae), (torch.randn(1, 4, h, h, dtype=dt),), check_trace=False)
    torch.jit.script(VaeBundle(enc, dec)).save(os.path.join(args.out_dir, "sd_vae.ts"))
    print(f"[sd15_faceid] done -> {args.out_dir}")


if __name__ == "__main__":
    main()

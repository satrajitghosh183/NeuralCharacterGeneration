#!/usr/bin/env python3
"""Refine the 5 lit portrait views with IP-Adapter-FaceID at a fixed strength, for reprojection bake.

Loads me_face/kj_face '<prefix>_face_lit_<+/-NN>.png' (az in -40,-20,0,20,40), refines each toward the
reference identity (FaceID, recognition embedding), writes '<out>/refine_<+/-NN>.png' — the input that
`ncg_cli face --reproject-dir <out>` bakes back onto the UV texture.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageFilter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lit-prefix", required=True, help="path prefix; loads <prefix>_face_lit_<az>.png")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--faceid-weight", default="ip-adapter-faceid-portrait_sd15.bin")
    ap.add_argument("--strength", type=float, default=0.8)
    ap.add_argument("--faceid-scale", type=float, default=1.0)
    ap.add_argument("--declean", type=int, default=7)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="a photo of a person, photorealistic skin texture, fine pores, "
                    "natural detailed skin, sharp, studio portrait")
    ap.add_argument("--neg", default="cartoon, cgi, 3d render, plastic, waxy, smooth, blurry, deformed")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    rp = Image.open(args.ref).convert("RGB")
    faces = app.get(np.asarray(rp)[:, :, ::-1].copy())
    if not faces:
        raise SystemExit("[faceid_views] no face in reference")
    e = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])).normed_embedding
    e = e / (np.linalg.norm(e) + 1e-9)
    pos = torch.from_numpy(e).to(device, dtype).reshape(1, 1, -1)
    faceid_embeds = torch.cat([torch.zeros_like(pos), pos], dim=0)  # [2,1,512]

    from diffusers import AutoPipelineForImage2Image, DPMSolverMultistepScheduler
    pipe = AutoPipelineForImage2Image.from_pretrained(
        args.model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    pipe.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None,
                         weight_name=args.faceid_weight, image_encoder_folder=None)
    pipe.set_ip_adapter_scale(args.faceid_scale)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass

    for az in (-40, -20, 0, 20, 40):
        src = f"{args.lit_prefix}_face_lit_{az:+03d}.png"
        if not os.path.exists(src):
            print(f"[faceid_views] MISSING {src}", flush=True)
            continue
        base = Image.open(src).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
        if args.declean >= 3:
            base = base.filter(ImageFilter.MedianFilter(args.declean | 1))
        gen = torch.Generator(device=device).manual_seed(args.seed)
        out = pipe(prompt=args.prompt, negative_prompt=args.neg, image=base, strength=args.strength,
                   ip_adapter_image_embeds=[faceid_embeds], num_inference_steps=args.steps,
                   guidance_scale=args.guidance, generator=gen).images[0]
        dst = os.path.join(args.out_dir, f"refine_{az:+03d}.png")
        out.save(dst)
        print(f"[faceid_views] {src} -> {dst}", flush=True)


if __name__ == "__main__":
    main()

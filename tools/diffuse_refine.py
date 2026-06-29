#!/usr/bin/env python3
"""Bake-time diffusion sidecar for CONFIDENCE-GATED PHOTOREAL REFINEMENT.

OFF-THE-SHELF ONLY: pretrained SD-1.5 (photoreal checkpoint) + pretrained IP-Adapter (face identity
conditioning from a reference image) + optional pretrained ControlNet (depth/normal, to pin the
recovered geometry). NO training / fine-tuning / LoRA. This runs at BAKE TIME (Python is fine here);
the shipped C++/CUDA runtime never sees diffusion.

The CONTRIBUTION is not "we ran SD" — it is that the recovered CONFIDENCE field o(v) gates where the
generative prior may act: high-confidence (well-observed) regions are REFINED at low denoising
strength (identity held tight); low-confidence (silent) regions are COMPLETED at higher strength.
We realize the per-pixel strength via DIFFERENTIAL DIFFUSION (a change-map controls how many
denoising steps each pixel receives) — see --gate-map. Without a gate map it falls back to a scalar
--strength (Phase-1 plumbing / sweep points).

Phases:
  --mode plumbing : refine ONE base image conditioned on a reference (proves the sidecar is wired).
  --mode refine   : refine a base render (+ optional control image + optional gate map) -> output.

Usage (Phase 1 GATE):
  python tools/diffuse_refine.py --mode plumbing --model <hf-id> \
      --base base.png --ref kendall.jpg --out gate1.png --strength 0.45 --ip-scale 0.6
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageFilter


def log(*a):
    print("[diffuse_refine]", *a, flush=True)


def load_image(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.LANCZOS)
    return img


def build_pipe(model, controlnet_id, ip_weight, dtype, device, diff_diffusion=False):
    """Load the off-the-shelf pipeline. ControlNet + IP-Adapter are both pretrained, downloaded."""
    from diffusers import (AutoPipelineForImage2Image, ControlNetModel,
                           DPMSolverMultistepScheduler,
                           StableDiffusionControlNetImg2ImgPipeline)
    kw = dict(torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    if controlnet_id:
        cn = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(model, controlnet=cn, **kw)
    else:
        # AutoPipeline wires the IP-Adapter attention processors correctly (the vanilla img2img
        # pipeline mis-routes the image-embed tuple to self-attention -> 'tuple' has no .shape).
        pipe = AutoPipelineForImage2Image.from_pretrained(model, **kw)
    # IP-Adapter: identity/appearance conditioning from a reference image (pretrained weights).
    # Empty ip_weight => skip (isolation / pure img2img). Loaded BEFORE to(device); diffusers pulls
    # the matching ViT-H image encoder from the same repo.
    if ip_weight:
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name=ip_weight)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    # NOTE: do NOT enable_attention_slicing() — it REPLACES the attention processors, clobbering the
    # IP-Adapter processors load_ip_adapter just installed (image-embed tuple then hits a vanilla
    # layer -> "'tuple' object has no attribute 'shape'"). VAE tiling is safe (VAE only).
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plumbing", "refine"], default="plumbing")
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5",
                    help="off-the-shelf SD-1.5 diffusers checkpoint (photoreal variant recommended)")
    ap.add_argument("--controlnet", default="", help="off-the-shelf ControlNet id (depth/normal); empty=off")
    ap.add_argument("--ip-weight", default="ip-adapter-plus-face_sd15.bin",
                    help="IP-Adapter weight in h94/IP-Adapter/models (face = identity conditioning)")
    ap.add_argument("--base", required=True, help="base render to refine (img2img init)")
    ap.add_argument("--ref", required=True, help="reference image for IP-Adapter identity conditioning")
    ap.add_argument("--control", default="", help="control image (depth/normal) for ControlNet")
    ap.add_argument("--gate-map", default="", help="per-pixel strength map (white=complete, black=keep); enables differential diffusion")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="a photo of a person, photorealistic skin texture, fine pores, "
                    "natural detailed skin, sharp, studio portrait, 85mm")
    ap.add_argument("--neg", default="cartoon, cgi, 3d render, plastic, waxy, smooth, blurry, "
                    "deformed, extra limbs, watermark, text")
    ap.add_argument("--strength", type=float, default=0.45, help="HIGH (completion) strength for low-conf regions / scalar strength")
    ap.add_argument("--gate-lo", type=float, default=0.25, help="LOW (refine) strength for confident regions")
    ap.add_argument("--declean", type=int, default=0, help="median-filter size to remove base texel-splat crack seams before refine (0=off)")
    ap.add_argument("--ip-scale", type=float, default=0.6)
    ap.add_argument("--control-scale", type=float, default=0.8)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=5.5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    log(f"device={device} dtype={dtype} model={args.model} controlnet={args.controlnet or 'none'} "
        f"ip={args.ip_weight}")

    pipe = build_pipe(args.model, args.controlnet, args.ip_weight, dtype, device)
    if args.ip_weight:
        pipe.set_ip_adapter_scale(args.ip_scale)

    base = load_image(args.base, args.size)
    if args.declean and args.declean >= 3:
        # The recovered albedo carries thin bright UV-seam cracks; a median filter removes them
        # (they're sub-structure spikes) without blurring real features, so refinement isn't seeded
        # with crack lines it then preserves. (A smoother C++ albedo render would obviate this.)
        base = base.filter(ImageFilter.MedianFilter(args.declean | 1))
        log(f"de-cracked base with median filter size {args.declean | 1}")
    ref_img = load_image(args.ref, args.size) if args.ip_weight else None
    ctrl_img = (load_image(args.control, args.size) if args.control else base) if args.controlnet else None

    def one_pass(strength, seed):
        gen = torch.Generator(device=device).manual_seed(seed)
        c = dict(prompt=args.prompt, negative_prompt=args.neg, image=base, strength=float(strength),
                 num_inference_steps=args.steps, guidance_scale=args.guidance, generator=gen)
        if ref_img is not None:
            c["ip_adapter_image"] = ref_img
        if ctrl_img is not None:
            c["control_image"] = ctrl_img
            c["controlnet_conditioning_scale"] = args.control_scale
        return pipe(**c).images[0]

    if args.gate_map:
        # CONFIDENCE-GATED refinement: refine confident regions at LOW strength (identity held) and
        # complete low-confidence regions at HIGH strength (invent), blended by the feathered
        # confidence = 1 - gate. Per-region strength that composes with IP-Adapter; the feather keeps
        # the refined<->completed boundary seamless.
        change = Image.open(args.gate_map).convert("L").resize((args.size, args.size), Image.LANCZOS)
        change = change.filter(ImageFilter.GaussianBlur(6))
        conf = 1.0 - np.asarray(change).astype(np.float32) / 255.0  # [H,W], 1=confident
        conf = conf[..., None]
        log(f"gated blend: lo(refine)={args.gate_lo} hi(complete)={args.strength} "
            f"mean_conf={float(conf.mean()):.3f}")
        lo = np.asarray(one_pass(args.gate_lo, args.seed)).astype(np.float32)
        hi = np.asarray(one_pass(args.strength, args.seed)).astype(np.float32)
        out = Image.fromarray(np.clip(conf * lo + (1.0 - conf) * hi, 0, 255).astype(np.uint8))
    else:
        log(f"scalar refine: strength={args.strength} ip_scale={args.ip_scale}")
        out = one_pass(args.strength, args.seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.save(args.out)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""PER-REGION confidence-driven FaceID refinement of the lit views (deterministic control law, no RL).

For each lit portrait view, blend a LOW-strength FaceID refine (gentle; preserves real features +
stays aligned to the mesh so it bakes cleanly) with a HIGH-strength FaceID refine (generative; for
sparse regions with no real detail), weighted by a per-pixel confidence map:
    confidence high  -> use LOW strength  (identity center: eyes/nose/mouth + well-observed front)
    confidence low   -> use HIGH strength (periphery / sparse)
    strength(px) realized as the blend lo*conf + hi*(1-conf).
Confidence = central-frontal falloff, with an IDENTITY-REGION LOCK boosting the landmark area
(eyes/nose/mouth via insightface kps) to ~1 so the recognizable center is always refined gently.
Output refine_<az>.png feeds `ncg_cli face --reproject-dir` for the bake.
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lit-prefix", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--faceid-weight", default="ip-adapter-faceid-portrait_sd15.bin")
    ap.add_argument("--s-lo", type=float, default=0.4, help="gentle strength (identity center)")
    ap.add_argument("--s-hi", type=float, default=0.75, help="generative strength (sparse periphery)")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--faceid-scale", type=float, default=1.0)
    ap.add_argument("--center-sigma", type=float, default=0.5)
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
    rfaces = app.get(np.asarray(rp)[:, :, ::-1].copy())
    if not rfaces:
        raise SystemExit("[faceid_region] no face in reference")
    e = max(rfaces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1])).normed_embedding
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

    def refine(base, strength):
        gen = torch.Generator(device=device).manual_seed(args.seed)
        return pipe(prompt=args.prompt, negative_prompt=args.neg, image=base, strength=float(strength),
                    ip_adapter_image_embeds=[faceid_embeds], num_inference_steps=args.steps,
                    guidance_scale=args.guidance, generator=gen).images[0]

    def confidence_map(base):
        # central-frontal falloff + identity-region lock (landmarks -> confidence ~1 -> gentle).
        W, H = base.size
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        s = args.center_sigma * max(W, H)
        conf = np.exp(-(((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (2 * s * s)))
        faces = app.get(np.asarray(base.convert("RGB"))[:, :, ::-1].copy())
        if faces:
            kps = max(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1])).kps  # [5,2]
            m = Image.new("L", (W, H), 0)
            d = ImageDraw.Draw(m)
            r = 0.16 * max(W, H)  # lock radius around each landmark
            for (px, py) in kps:
                d.ellipse([px - r, py - r, px + r, py + r], fill=255)
            lock = np.asarray(m.filter(ImageFilter.GaussianBlur(r * 0.4))).astype(np.float32) / 255.0
            conf = np.maximum(conf, lock)  # identity landmarks -> confident -> gentle
        return np.clip(conf, 0.0, 1.0) ** args.gamma

    for az in (-40, -20, 0, 20, 40):
        src = f"{args.lit_prefix}_face_lit_{az:+03d}.png"
        if not os.path.exists(src):
            print(f"[faceid_region] MISSING {src}", flush=True)
            continue
        base = Image.open(src).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
        if args.declean >= 3:
            base = base.filter(ImageFilter.MedianFilter(args.declean | 1))
        conf = confidence_map(base)[..., None]                       # [H,W,1] 1=confident
        lo = np.asarray(refine(base, args.s_lo)).astype(np.float32)
        hi = np.asarray(refine(base, args.s_hi)).astype(np.float32)
        blend = conf * lo + (1.0 - conf) * hi                        # gentle center, strong periphery
        Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8)).save(
            os.path.join(args.out_dir, f"refine_{az:+03d}.png"))
        if az == 0:
            Image.fromarray((conf[..., 0] * 255).astype(np.uint8)).save(
                os.path.join(args.out_dir, "confidence_+00.png"))
        print(f"[faceid_region] {src} -> per-region blend (mean conf {float(conf.mean()):.2f})", flush=True)


if __name__ == "__main__":
    main()

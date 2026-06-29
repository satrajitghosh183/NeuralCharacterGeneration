#!/usr/bin/env python3
"""RECOGNITION-GRADE conditioner test: IP-Adapter-FaceID (portrait variant, no LoRA) refinement sweep.

Unlike IP-Adapter-plus-face (CLIP image features), FaceID conditions generation on the reference's
ArcFace FACE-RECOGNITION embedding (insightface) — directly targeting identity. Off-the-shelf
released weights; no training/LoRA (the -portrait variant is embedding-only). Bake-time.

Sweeps img2img strength at a fixed FaceID scale; at each point measures ArcFace(refined, ref)
identity + CLIP-IQA realism. Same harness/metrics as tools/tradeoff.py so curves overlay directly.

NOTE: conditioner and identity metric share the ArcFace embedding family, so a rise in identity is
EXPECTED *if the conditioning takes* — that is exactly the question (can recognition-grade
conditioning inject identity that CLIP-based IP-Adapter could not, onto these bases?). Realism is
independent.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageFilter


def log(*a):
    print("[faceid]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--faceid-weight", default="ip-adapter-faceid-portrait_sd15.bin")
    ap.add_argument("--strengths", default="0.2,0.35,0.5,0.65,0.8,0.95")
    ap.add_argument("--faceid-scale", type=float, default=1.0)
    ap.add_argument("--declean", type=int, default=7)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prompt", default="a photo of a person, photorealistic skin texture, fine pores, "
                    "natural detailed skin, sharp, studio portrait, 85mm")
    ap.add_argument("--neg", default="cartoon, cgi, 3d render, plastic, waxy, smooth, blurry, deformed")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    # ---- insightface: ref ArcFace embed (conditioning) + scorer ----
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    def get_embed(pil, pad):
        pil = pil.convert("RGB")
        if pad:
            w, h = pil.size
            p = int(0.6 * max(w, h))
            c = Image.new("RGB", (w + 2 * p, h + 2 * p), (128, 128, 128))
            c.paste(pil, (p, p))
            pil = c
        faces = app.get(np.asarray(pil)[:, :, ::-1].copy())
        if not faces:
            return None
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        e = f.normed_embedding
        return e / (np.linalg.norm(e) + 1e-9)

    ref_pil = Image.open(args.ref).convert("RGB")
    ref_e = get_embed(ref_pil, pad=False)
    if ref_e is None:
        raise SystemExit("[faceid] no face in reference; cannot condition")
    # FaceID conditioning embed must be pre-stacked [negative; positive] -> [2,1,512]; diffusers
    # chunk(2)s it into uncond/cond. Negative = zeros.
    pos = torch.from_numpy(ref_e).to(device, dtype).reshape(1, 1, -1)  # [1,1,512]
    faceid_embeds = torch.cat([torch.zeros_like(pos), pos], dim=0)     # [2,1,512]
    log(f"reference ArcFace embed ready {tuple(faceid_embeds.shape)}")

    # ---- pipeline + FaceID adapter ----
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

    base = Image.open(args.base).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
    if args.declean >= 3:
        base = base.filter(ImageFilter.MedianFilter(args.declean | 1))
    base.save(os.path.join(args.out_dir, "base_declean.png"))

    def refine(strength):
        gen = torch.Generator(device=device).manual_seed(args.seed)
        return pipe(prompt=args.prompt, negative_prompt=args.neg, image=base, strength=float(strength),
                    ip_adapter_image_embeds=[faceid_embeds], num_inference_steps=args.steps,
                    guidance_scale=args.guidance, generator=gen).images[0]

    strengths = [float(s) for s in args.strengths.split(",")]
    renders = {0.0: base}
    for s in strengths:
        log(f"faceid refine strength={s} scale={args.faceid_scale}")
        img = refine(s)
        img.save(os.path.join(args.out_dir, f"refine_{s:.2f}.png"))
        renders[s] = img

    del pipe
    torch.cuda.empty_cache()

    iqa = None
    try:
        import pyiqa
        iqa = pyiqa.create_metric("clipiqa", device="cpu")
    except Exception as e:
        log(f"WARN pyiqa: {e}")

    def realism(pil):
        if iqa is None:
            return None
        t = torch.from_numpy(np.asarray(pil.convert("RGB")).copy()).permute(2, 0, 1)[None].float() / 255.0
        return float(iqa(t).item())

    rows = []
    for s in sorted(renders):
        re = get_embed(renders[s], pad=True)
        idv = float(np.dot(re, ref_e)) if re is not None else None
        rev = realism(renders[s])
        rows.append({"strength": s, "identity_arcface": idv, "realism_clipiqa": rev})
        log(f"  s={s:.2f} identity={idv} realism={rev}")
    with open(os.path.join(args.out_dir, "metrics.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    base_id = rows[0]["identity_arcface"]
    log(f"base identity={base_id}; max identity over sweep="
        f"{max((r['identity_arcface'] for r in rows[1:] if r['identity_arcface'] is not None), default=None)}")


if __name__ == "__main__":
    main()

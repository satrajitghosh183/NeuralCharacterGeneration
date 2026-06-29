#!/usr/bin/env python3
"""IDENTITY-vs-REALISM TRADEOFF SWEEP — the Phase-3 deliverable.

Refines a face base across a sweep of diffusion conditioning STRENGTH (the primary knob), and at each
point measures BOTH:
  identity  = ArcFace cosine(refined, real reference)   [off-the-shelf insightface]
  realism   = a no-reference image-quality score        [off-the-shelf pyiqa, CLIP-IQA]
Plots the curve and reports the KNEE: the most photoreal point whose identity stays >= the base's
ArcFace identity (the IDENTITY GUARD — photoreal-but-wrong-person is a failure). Every point dumps its
render so the curve is backed by images, not just numbers.

Off-the-shelf only; bake-time. No training.
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image, ImageFilter


def log(*a):
    print("[tradeoff]", *a, flush=True)


def build_pipe(model, ip_weight, dtype, device):
    from diffusers import AutoPipelineForImage2Image, DPMSolverMultistepScheduler
    pipe = AutoPipelineForImage2Image.from_pretrained(
        model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
    if ip_weight:
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name=ip_weight)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    try:
        pipe.enable_vae_tiling()  # NOT attention_slicing (clobbers IP-adapter processors)
    except Exception:
        pass
    return pipe


class ArcScorer:
    """Off-the-shelf ArcFace identity via insightface; cosine of L2-normed embeddings."""
    def __init__(self):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def embed(self, pil):
        # Pad a margin: face detectors fail when the face FILLS the frame (our portraits do). Pasting
        # into a larger gray canvas puts the face at a moderate scale the detector was trained on.
        pil = pil.convert("RGB")
        w, h = pil.size
        pad = int(0.6 * max(w, h))
        canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (128, 128, 128))
        canvas.paste(pil, (pad, pad))
        bgr = np.asarray(canvas)[:, :, ::-1].copy()
        faces = self.app.get(bgr)
        if not faces:
            return None
        f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        e = f.normed_embedding
        return e / (np.linalg.norm(e) + 1e-9)

    def cos(self, a, b):
        ea, eb = self.embed(a), self.embed(b)
        if ea is None or eb is None:
            return None
        return float(np.dot(ea, eb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--ip-weight", default="ip-adapter-plus-face_sd15.bin")
    ap.add_argument("--strengths", default="0.2,0.35,0.5,0.65,0.8,0.95")
    ap.add_argument("--sweep-ip", default="", help="if set (comma list), sweep IP-adapter scale at fixed --strength instead of sweeping strength")
    ap.add_argument("--strength", type=float, default=0.6, help="fixed strength when --sweep-ip is used")
    ap.add_argument("--ip-scale", type=float, default=0.85)
    ap.add_argument("--declean", type=int, default=7)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--guidance", type=float, default=5.5)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--score-only", action="store_true", help="skip generation; score existing renders in out-dir")
    ap.add_argument("--prompt", default="a photo of a person, photorealistic skin texture, fine pores, "
                    "natural detailed skin, sharp, studio portrait, 85mm")
    ap.add_argument("--neg", default="cartoon, cgi, 3d render, plastic, waxy, smooth, blurry, deformed")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    ref = Image.open(args.ref).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
    # Sweep variable: STRENGTH (default), or IP-ADAPTER SCALE if --sweep-ip given (fixed strength).
    sweep_ip = [float(s) for s in args.sweep_ip.split(",")] if args.sweep_ip else None
    sweep_vals = sweep_ip if sweep_ip is not None else [float(s) for s in args.strengths.split(",")]
    sweep_name = "ip_scale" if sweep_ip is not None else "strength"
    strengths = sweep_vals  # for score-only file naming compatibility

    if args.score_only:
        renders = {0.0: Image.open(os.path.join(args.out_dir, "base_declean.png")).convert("RGB")}
        for s in strengths:
            renders[s] = Image.open(os.path.join(args.out_dir, f"refine_{s:.2f}.png")).convert("RGB")
        log(f"score-only: loaded {len(renders)} existing renders")
    else:
        base = Image.open(args.base).convert("RGB").resize((args.size, args.size), Image.LANCZOS)
        if args.declean >= 3:
            base = base.filter(ImageFilter.MedianFilter(args.declean | 1))
        base.save(os.path.join(args.out_dir, "base_declean.png"))
        pipe = build_pipe(args.model, args.ip_weight, dtype, device)
        if args.ip_weight:
            pipe.set_ip_adapter_scale(args.ip_scale)

        def refine(val):
            # val is the swept variable: strength (default) or ip-scale (--sweep-ip).
            strength = args.strength if sweep_ip is not None else float(val)
            if sweep_ip is not None and args.ip_weight:
                pipe.set_ip_adapter_scale(float(val))
            gen = torch.Generator(device=device).manual_seed(args.seed)
            c = dict(prompt=args.prompt, negative_prompt=args.neg, image=base, strength=float(strength),
                     num_inference_steps=args.steps, guidance_scale=args.guidance, generator=gen)
            if args.ip_weight:
                c["ip_adapter_image"] = ref
            return pipe(**c).images[0]

        renders = {0.0: base}  # 0 = the (de-cleaned) base itself
        for s in sweep_vals:
            log(f"refining {sweep_name}={s}")
            img = refine(s)
            img.save(os.path.join(args.out_dir, f"refine_{s:.2f}.png"))
            renders[s] = img
        del pipe
        torch.cuda.empty_cache()
    arc = None
    try:
        arc = ArcScorer()
    except Exception as e:
        log(f"WARN insightface unavailable: {e}")
    iqa = None
    try:
        import pyiqa
        iqa = pyiqa.create_metric("clipiqa", device="cpu")
    except Exception as e:
        log(f"WARN pyiqa unavailable: {e}")

    def realism(pil):
        if iqa is None:
            return None
        t = torch.from_numpy(np.asarray(pil.convert("RGB"))).permute(2, 0, 1)[None].float() / 255.0
        return float(iqa(t).item())

    rows = []
    for s in sorted(renders):
        idv = arc.cos(renders[s], ref) if arc else None
        rev = realism(renders[s])
        rows.append({"strength": s, "identity_arcface": idv, "realism_clipiqa": rev})
        log(f"  s={s:.2f}  identity={idv}  realism={rev}")
    with open(os.path.join(args.out_dir, "metrics.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # IDENTITY GUARD: knee = max realism among points with identity >= base identity.
    base_id = rows[0]["identity_arcface"]
    knee = None
    if base_id is not None:
        cand = [r for r in rows if r["strength"] > 0 and r["identity_arcface"] is not None
                and r["identity_arcface"] >= base_id and r["realism_clipiqa"] is not None]
        if cand:
            knee = max(cand, key=lambda r: r["realism_clipiqa"])
    log(f"base identity={base_id}; KNEE (max realism with identity>=base) = {knee}")

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r["strength"] for r in rows if r["identity_arcface"] is not None and r["realism_clipiqa"] is not None]
        idv = [r["identity_arcface"] for r in rows if r["identity_arcface"] is not None and r["realism_clipiqa"] is not None]
        rev = [r["realism_clipiqa"] for r in rows if r["identity_arcface"] is not None and r["realism_clipiqa"] is not None]
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax1.set_xlabel("diffusion strength"); ax1.set_ylabel("identity (ArcFace cos)", color="tab:blue")
        ax1.plot(xs, idv, "o-", color="tab:blue", label="identity")
        if base_id is not None:
            ax1.axhline(base_id, ls="--", color="tab:blue", alpha=0.5, label="base identity (guard)")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax2 = ax1.twinx(); ax2.set_ylabel("realism (CLIP-IQA)", color="tab:red")
        ax2.plot(xs, rev, "s-", color="tab:red", label="realism")
        ax2.tick_params(axis="y", labelcolor="tab:red")
        if knee:
            ax1.axvline(knee["strength"], ls=":", color="green", label=f"knee s={knee['strength']:.2f}")
        fig.suptitle("Identity vs Realism tradeoff (confidence-gated off-the-shelf refinement)")
        fig.legend(loc="lower center", ncol=4, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "tradeoff_curve.png"), dpi=120)
        log(f"wrote {os.path.join(args.out_dir, 'tradeoff_curve.png')}")
    except Exception as e:
        log(f"WARN plot failed: {e}")


if __name__ == "__main__":
    main()

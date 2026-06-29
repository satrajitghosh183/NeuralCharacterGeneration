#!/usr/bin/env python3
"""Stage 3: the two-base x two-conditioner identity contrast (the scientific statement).

Overlays ArcFace identity vs diffusion strength for {Kendall, user} x {IP-Adapter-plus-face (CLIP),
IP-Adapter-FaceID (recognition embedding)}, reading each run's metrics.jsonl. Shows that identity
injection tracks the CONDITIONER (recognition-grade) — CLIP-features floor on both bases, FaceID
injects on both — localizing identity to the conditioning, with the base providing structure.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(run):
    xs, ids = [], []
    p = os.path.join(run, "metrics.jsonl")
    if not os.path.exists(p):
        return xs, ids
    for line in open(p):
        r = json.loads(line)
        if r.get("identity_arcface") is not None:
            xs.append(r["strength"]); ids.append(r["identity_arcface"])
    return xs, ids


series = [
    ("runs/kj_tradeoff", "Kendall · plus-face (CLIP)", "tab:blue", ":"),
    ("runs/kj_faceid", "Kendall · FaceID (recognition)", "tab:blue", "-"),
    ("runs/me_tradeoff", "user · plus-face (CLIP)", "tab:red", ":"),
    ("runs/me_faceid", "user · FaceID (recognition)", "tab:red", "-"),
]
fig, ax = plt.subplots(figsize=(8, 5.5))
for run, label, color, ls in series:
    xs, ids = load(run)
    if xs:
        ax.plot(xs, ids, marker="o", color=color, ls=ls, label=label)
ax.axhline(0.325, color="gray", ls="--", alpha=0.6, label="Kendall same-person (0.325)")
ax.axhline(0.746, color="black", ls="--", alpha=0.6, label="user same-person (0.746)")
ax.set_xlabel("diffusion strength")
ax.set_ylabel("identity — ArcFace cos(refined, real reference)")
ax.set_title("Identity injection tracks the CONDITIONER, not the base\n"
             "(CLIP-features floor on both bases; recognition-embedding injects on both)")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "runs/identity_contrast.png"
fig.savefig(out, dpi=130)
print("wrote", out, flush=True)

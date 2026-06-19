#!/usr/bin/env python3
"""Export a PyTorch model's state_dict to safetensors with provenance metadata.

DEV-ONLY: runs on a workstation that has the reference model + its repo. The produced
.safetensors is consumed by the C++ WeightMap (ncg/io/weight_loader.hpp). See docs/parity.md.

This is intentionally a thin, generic helper. For each model we add a small loader function
in MODEL_LOADERS that returns (state_dict, provenance_dict). Keeping the load logic here (not
in the C++ tree) preserves the "no Python at runtime" rule.
"""
from __future__ import annotations

import argparse
import sys

import torch
from safetensors.torch import save_file


def _load_nlf():
    """Return (state_dict, provenance) for NLF. Fill in once the repo is vendored locally."""
    raise SystemExit(
        "export_weights: NLF loader not configured yet. Clone isarandi/nlf, load its\n"
        "checkpoint here, and return (model.state_dict(), {'repo': ..., 'commit': ...})."
    )


MODEL_LOADERS = {
    "nlf": _load_nlf,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_LOADERS))
    ap.add_argument("--out", required=True, help="output .safetensors path")
    args = ap.parse_args()

    state_dict, provenance = MODEL_LOADERS[args.model]()

    # safetensors requires contiguous CPU tensors; metadata values must be strings.
    tensors = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    metadata = {"__ncg__": "exported by tools/export_weights.py"}
    metadata.update({k: str(v) for k, v in provenance.items()})

    save_file(tensors, args.out, metadata=metadata)
    print(f"wrote {len(tensors)} tensors -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

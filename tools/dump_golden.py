#!/usr/bin/env python3
"""Dump per-submodule golden activations for C++ parity tests.

DEV-ONLY (workstation). Registers forward hooks on every named submodule of the reference
model, runs a fixed seeded input, and writes input.npy, <stage>.npy per checked submodule,
and manifest.json into the output dir. The C++ test (tests/golden/test_golden_<m>.cpp) then
asserts the ported forward matches each stage. See docs/parity.md.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import torch


def dump(model: torch.nn.Module, example_input: torch.Tensor, out_dir: str,
         stage_modules: list[str], weights_name: str, provenance: dict,
         rtol: float = 1e-3, atol: float = 1e-4) -> None:
    import os

    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    torch.manual_seed(0)
    # Match the C++ parity contract: deterministic fp32, no TF32.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    captured: dict[str, torch.Tensor] = {}
    handles = []
    wanted = set(stage_modules)
    for name, module in model.named_modules():
        if name in wanted:
            def hook(_m, _inp, out, _name=name):
                t = out[0] if isinstance(out, (tuple, list)) else out
                captured[_name] = t.detach().cpu()
            handles.append(module.register_forward_hook(hook))

    with torch.no_grad():
        output = model(example_input)
    for h in handles:
        h.remove()

    np.save(os.path.join(out_dir, "input.npy"), example_input.detach().cpu().numpy())

    stages = []
    for name in stage_modules:
        if name not in captured:
            raise SystemExit(f"dump_golden: submodule '{name}' produced no output")
        fname = name.replace(".", "_") + ".npy"
        np.save(os.path.join(out_dir, fname), captured[name].numpy())
        stages.append({"name": name, "ref": fname})

    out_t = output[0] if isinstance(output, (tuple, list)) else output
    np.save(os.path.join(out_dir, "output.npy"), out_t.detach().cpu().numpy())
    stages.append({"name": "output", "ref": "output.npy"})

    manifest = {
        "model": os.path.basename(out_dir.rstrip("/")),
        "weights": weights_name,
        "input": "input.npy",
        "source": provenance,
        "tolerance": {"rtol": rtol, "atol": atol},
        "stages": stages,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(stages)} stages + manifest -> {out_dir}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.parse_args()
    raise SystemExit(
        "dump_golden: per-model setup required. Import the reference model, build an example\n"
        "input, choose the stage_modules to check, and call dump(...). See docstring."
    )


if __name__ == "__main__":
    sys.exit(main())

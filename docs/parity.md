# Weight porting & numerical parity

How we port a published PyTorch model into C++/CUDA **without retraining** and prove the port
is correct. This is the project's central de-risker (see `docs/plan.md` §6).

## Principle
Reimplement the architecture in C++ (LibTorch + custom kernels), **load the published
weights**, and assert the C++ forward pass matches the reference **layer by layer**. The
per-layer check localizes any divergence to a single module instead of "output is wrong."

## Format
We use **safetensors** directly (no custom binary): `[u64 header_len][JSON header][bytes]`.
Reader/writer: `ncg/io/safetensors.hpp`. Loader with name remap + completeness asserts:
`ncg/io/weight_loader.hpp` (`WeightMap`).

## Workflow to add a model `<m>`
1. **Export weights** (offline, on a workstation with the reference repo):
   ```bash
   python tools/export_weights.py --model <m> --out data/golden/<m>/<m>.safetensors
   ```
   Re-saves the `state_dict` as safetensors and injects provenance (`__ncg__` metadata:
   source repo + commit).
2. **Dump golden activations** with forward hooks on every named submodule:
   ```bash
   python tools/dump_golden.py --model <m> --out data/golden/<m>
   ```
   Produces `input.npy`, one `<stage>.npy` per checked submodule, and `manifest.json`:
   ```json
   {
     "model": "<m>", "weights": "<m>.safetensors", "input": "input.npy",
     "source": {"repo": "...", "commit": "..."},
     "tolerance": {"rtol": 1e-3, "atol": 1e-4},
     "stages": [
       {"name": "backbone", "ref": "backbone.npy"},
       {"name": "head", "ref": "head.npy", "rtol": 2e-3},
       {"name": "output", "ref": "output.npy"}
     ]
   }
   ```
3. **C++ side**: expose the same intermediates (a debug forward returning named stages, or
   call the submodules individually). Add `tests/golden/test_golden_<m>.cpp` using
   `golden_fixture.hpp` + `tensor_compare.hpp`; iterate stages in order and report the FIRST
   failing one.
4. The test **SKIPs** when `data/golden/<m>/manifest.json` is absent (large dumps/weights are
   fetched via git-LFS, not committed by default), so CI stays green on a fresh checkout.

## Tolerance policy
- Default `rtol=1e-3, atol=1e-4` for fp32 forward parity.
- conv/attention accumulation order differs from cuDNN → relative error grows with depth;
  set per-stage tolerances in the manifest rather than loosening the global one.
- **Disable TF32** before parity runs: the harness calls `ncg::set_deterministic_fp32(true)`.
  On Hopper, TF32 (on by default) silently exceeds fp32 tolerances and fakes a port bug.
- Seed everything; record the seed + flags in the manifest.

## Triage order when a real golden test fails
1. Run `ctest -R golden_linear` — if the harness self-test fails, the bug is in the harness
   / loader, not the port.
2. Find the first failing stage in `test_golden_<m>` → that submodule is where it diverged.
3. Common causes: transposed/!contiguous weight, wrong name remap, missing buffer
   (running stats / positional encodings), TF32, an off-by-one in a custom kernel.

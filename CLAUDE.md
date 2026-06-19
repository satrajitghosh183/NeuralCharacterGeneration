# CLAUDE.md — NeuralCharGen

Working guide for this repo. Kept current as the project evolves (see "Working discipline").

## What this is
A system that turns a few **casual personal photos** into a **hyperrealistic, rigged, relightable, full-body 3D avatar** deployable to **Unity and Unreal**. Built **entirely in C++/CUDA** (no Python at runtime or train time). Research target: a SIGGRAPH paper. Full architecture, novelty, and roadmap live in `docs/plan.md`; the per-stage SOTA decisions in `docs/research-findings.md`.

## Golden rules
- **Port, don't retrain.** Reimplement published model architectures in C++/CUDA but **load their pretrained weights** and **prove forward-pass numerical parity** (see `docs/parity.md`). Train from scratch *only* the novel modules (fusion, selector, runtime).
- **Build-blind discipline.** Code is authored on a Mac with **no CUDA**; it is built/run on the **H100 Linux box (sm_90)**. So: small translation units, every custom kernel ships with a "vs LibTorch reference" test, every ported model has a **per-layer** golden test, and TF32 is disabled in parity runs.
- **Baseline first.** Get an unbroken end-to-end spine working before optimizing or adding novelty.

## Layout
```
ncg-core/      tensor/device helpers (LibTorch), cuda checks, logging, custom-kernel pattern
ncg-io/        image I/O, safetensors loader, WeightMap, .npy I/O   (Phase 0 part 2)
ncg-select/    bad-upload quality filtering + learned view selector (Phase 1 stub → P3)
ncg-body/      SMPL-X eval + NLF regressor (image → SMPL-X params)  (Phase 1)
ncg-recon/     Gaussian avatar + cross-photo fusion; 3DGS rasterizer (Phase 1 → P2/P3)
ncg-material/  intrinsic decomposition / delighting → PBR           (skeleton → P4)
ncg-rig/       skinning + skeleton; retarget to Unity/Unreal        (skeleton → later)
ncg-mesh/      mesh extraction + PBR bake                           (skeleton → P4)
ncg-runtime/   renderer: forward-splat CUDA kernel + differentiable soft renderer (Phase 1 → P2)
ncg-fit/       Gaussian optimization loop (fit cloud to image via autograd + Adam)  (Phase 2)
ncg-record/    experiment recording (run dirs, metrics.jsonl, image dumps, timers) + PSNR/SSIM/MAE
ncg-unreal/    Unreal native integration                           (P2+)
ncg-unity/     Unity native plugin                                 (P4+)
apps/          ncg_viewer (headless render), ncg_cli (pipeline)
tests/         unit + cuda-kernel + golden parity tests (Catch2 + CTest)
cmake/         NcgAddLibrary, NcgAddTest, NcgCudaArch, NcgWarnings
third_party/   FetchContent: spdlog, nlohmann_json, Catch2, stb, eigen
tools/         Mac/workstation Python: export_weights.py, dump_golden.py (DEV ONLY)
docs/          plan.md (living roadmap), research-findings.md, build.md, parity.md
```
Public headers are included as `#include <ncg/<module>/<file>.hpp>`. `.cu` files live in `kernels/` subdirs and never appear in public headers.

## Build / run / test (on the H100)
Prereqs: CUDA 12.x, LibTorch (cxx11-ABI) at `$LIBTORCH_ROOT`, CMake ≥3.27, Ninja, ccache. See `docs/build.md` for exact setup + version pins.
```bash
export LIBTORCH_ROOT=/path/to/libtorch
cmake --preset h100-release
cmake --build --preset h100-release
ctest --preset h100                 # all tests
ctest --preset h100 -L golden       # parity tests only
ctest --preset h100 -L cuda         # custom-kernel tests only
# tight loop: build one target
cmake --build --preset h100-release --target test_elementwise_kernel
# kernel fault triage:
compute-sanitizer ./build/h100-release/tests/core/test_elementwise_kernel
```
`scripts/ci.sh` runs configure→build→ctest in one shot (added with Phase 0 part 2).

Mac-side (lint only, optional, needs local CPU LibTorch): `cmake --preset mac-lint` to emit `compile_commands.json`, then `clang-tidy -p build/mac-lint <file.cpp>`. Always `clang-format -i` before pushing.

## Conventions
- C++20. Namespace `ncg`. Errors via `NCG_THROW` / `NCG_CHECK` (`ncg/core/error.hpp`).
- CUDA: `NCG_CUDA_CHECK(call)` and `NCG_CUDA_KERNEL_CHECK()` after every launch (`ncg/core/cuda_check.hpp`). Debug builds sync-check by default; toggle with env `NCG_CUDA_SYNC=0|1`.
- Logging: `NCG_LOG_INFO(...)` etc.; level via env `NCG_LOG_LEVEL`.
- Custom kernels: Tier-1 plain C++ entry points by default; Tier-2 (`TORCH_LIBRARY` + autograd Function) only when a kernel needs gradients (see `ncg/core/kernel_registry.hpp`).
- Weights: safetensors format (no custom binary). `WeightMap` asserts every param is filled and every entry consumed.

## Working discipline (keep these current)
- When a unit lands and is green on the H100: tick the phase in **`docs/plan.md`** and update **this file** if commands/layout/conventions changed.
- Dev loop is **git push → user builds on H100 → pastes back errors**. The user provides the git remote.

## Status
Phase 0 + Phase 1 scaffold + data-recording authored (full build, **pending first H100 build/test**). Implemented: build system, `ncg-core`, `ncg-io` (image/safetensors/WeightMap/npy), parity harness (`test_golden_linear` self-test), `ncg-body` SMPL-X LBS forward, `ncg-recon`, `ncg-runtime` (camera + forward-splat CUDA kernel), `ncg-select` (sharpness), `ncg-record` (run dirs + metrics.jsonl + image dumps + timers + PSNR/SSIM/MAE, wired into the viewer so every stage records), `apps/ncg_viewer` + `apps/ncg_cli`, tests, `tools/` export/dump, `scripts/ci.sh`. **Stubs/skeletons:** `ncg::body::Nlf` (load/predict throw — first real port target, see `docs/parity.md`), `ncg-material`/`ncg-rig`/`ncg-mesh`. The slice renders the neutral SMPL-X body (NLF image→pose not yet wired). Build/test module-by-module via `scripts/ci.sh`; see `docs/plan.md` progress log.

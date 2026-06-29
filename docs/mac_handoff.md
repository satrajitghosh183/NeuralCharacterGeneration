# Mac handoff — developing NeuralCharGen without the H100

When the H100 lease ends, development continues on the Mac. This is the canonical guide for what
builds/runs locally, the exact commands, and what stays H100-only. Validated 2026-06-29:
Apple clang 21.0.0, CMake 4.3.4, pip PyTorch 2.8.0 (CPU LibTorch), macOS arm64.

## TL;DR — local CPU build + test
```bash
TORCH=$(python3 -c 'import torch,os;print(os.path.dirname(torch.__file__))')   # pip torch = CPU LibTorch
cmake -G Ninja -B build/mac-cpu -S . \
  -DCMAKE_PREFIX_PATH="$TORCH" \
  -DNCG_BUILD_APPS=OFF -DNCG_BUILD_TESTS=ON -DNCG_LINT_ONLY=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/mac-cpu
ctest --test-dir build/mac-cpu            # -> 100% pass, 3 by-design skips (asset/CUDA-gated)
```
No CUDA toolkit is needed: with no `nvcc`, the top-level CMake sets `NCG_WITH_CUDA OFF`, every `.cu`
source is dropped (`NcgAddLibrary`), and CUDA-only libraries/tests are gated out (below). The pip
`torch` wheel ships a CPU LibTorch whose CMake config `find_package(Torch)` picks up via
`CMAKE_PREFIX_PATH`.

## What builds + tests on the Mac (CPU)
The CPU-clean module subset — all pure LibTorch/Eigen/stb, no custom CUDA:
`ncg-core, ncg-io, ncg-body, ncg-recon, ncg-geom, ncg-diffuse, ncg-material, ncg-rig, ncg-mesh,
ncg-record, ncg-select`. Their unit + golden tests run locally (58 tests, 100% pass, 3 skips).

This covers, on CPU: tensor/device/error/logging core; image + safetensors + WeightMap + npy I/O;
SMPL-X LBS forward + SMPL-X math; the recon inverse-render solver (C1/C2/relight); marching-cubes +
OBJ/PLY/glTF mesh export (incl. `write_glb`, `write_gaussian_ply`); analytic relight; rig
inheritance; the geometry Δv/observability solver; the ncg-diffuse SDS/DDPM/completion CPU math;
the recorder + PSNR/SSIM/MAE; the selector. **Most of the paper's math is locally
iterable+testable.**

## What is H100-only (CUDA) — and how to extend it to CPU later
Gated behind `NCG_WITH_CUDA` (top-level `CMakeLists.txt` + `tests/CMakeLists.txt`):
`ncg-runtime, ncg-fit, ncg-nerf`, the `apps/` (`ncg_cli`, `ncg_viewer`), and their tests.

Why: `ncg-runtime/src/renderer.cpp::render_gaussians` calls the `splat_forward.cu` kernel
(`splat_render_cuda`), and `ncg-fit` / `ncg-nerf` / the apps build on that renderer. The kernel
itself is CUDA-only; everything that *uses* it is pulled in with it.

To make `ncg-runtime/fit/nerf` build on CPU (a future increment, ~half a day):
1. Guard `render_gaussians` (the fast forward splatter) with `#if NCG_WITH_CUDA` and provide a CPU
   fallback or a clear runtime throw — the differentiable `render_soft` / `render_soft_aniso` paths
   are already pure LibTorch and run on CPU.
2. Move `add_subdirectory(ncg-runtime/fit/nerf)` out of the `if(NCG_WITH_CUDA)` block (top-level)
   and drop the matching guard in `tests/CMakeLists.txt`.
3. Relax `if(NCG_BUILD_APPS AND NCG_WITH_CUDA)` so `apps/` builds when its CPU render path exists.
CPU fits will be SLOW (the soft renderer is O(N·H·W)); use small N/res for local iteration.

NLF (`ncg-body/src/nlf.cpp`) loads a TorchScript graph that ran on CUDA on the H100; CPU inference
is possible but slow and the released trace may have a device baked in — treat NLF + large fits as
H100/GPU work.

## Toolchain notes / gotchas (already fixed in-tree)
- **fmt consteval (Apple clang).** spdlog v1.15.0 bundles fmt 11.0.2, whose `consteval` compile-time
  format-string check miscompiles on modern Apple clang. An APPLE-only `PATCH_COMMAND` in
  `third_party/CMakeLists.txt` neuters `FMT_CONSTEVAL`. The H100/GCC build is byte-identical.
- **CTest skips.** A `SKIP()`-ed Catch2 case exits 4; `NcgAddTest.cmake` sets
  `SKIP_REGULAR_EXPRESSION "SKIPPED:"` (before `LABELS`, so a multi-label split can't corrupt it) so
  gated tests record as Skipped, not Failed, on both platforms.
- **clang-format** isn't installed locally (`brew install clang-format`); run before pushing.
- `tools/*.py` are Mac/dev-side weight-export + diagnostics (need a venv with torch/diffusers/etc.);
  they are NOT part of the C++ runtime.

## Dev loop after the H100
Author + `build/mac-cpu` build + `ctest` locally on the CPU-clean subset. For the CUDA renderer /
fit / NLF / apps, you need a CUDA Linux box: re-provision an H100 (or any sm_80+), set
`LIBTORCH_ROOT`, `cmake --preset h100-release`, then the usual `git push -> reset -> build -> run`.

## Deliverables already pulled local
`~/Downloads/NCG_deliverables/` holds the latest avatars: `kjba.{glb,ply,ply.skin}` +
`kjba_turn0*.png` (turntable) + `kjba_fit0.png`, `kjw.{glb,ply,ply.skin}`, and the
`*.ply.skin` rigs for `rock_char` / `me_clean`. `.glb` = rigged 55-joint SMPL-X mesh (any
engine); `.ply` = 3DGS splats (Inria convention); `.ply.skin` = per-splat top-4 bone weights for
in-engine GS skinning. See `docs/engine_character.md`.

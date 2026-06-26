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
ncg-material/  analytical relight (real) + intrinsic decomposition (IDArb gated → P4)
ncg-rig/       rig inheritance + OBJ/JSON export (real); UniRig + FBX/glTF gated
ncg-mesh/      marching-cubes mesh extraction + OBJ/PLY export      (PBR bake → P4)
ncg-runtime/   renderer: forward-splat CUDA kernel + differentiable soft renderer (Phase 1 → P2)
ncg-fit/       Gaussian optimization loop (fit cloud to image via autograd + Adam)  (Phase 2)
ncg-nerf/      TinyNerf implicit volume + differentiable volume render + hybrid composite_over (NeRF leg)
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
**Canonical overview: `docs/architecture.md`** (system diagram + module map + the unifying thesis). Math: `docs/method.md`. Stats: `docs/benchmarks.md`. Roadmap/log: `docs/plan.md`.

**PHOTOREAL RECONSTRUCTION LEG GREEN on H100** (2026-06-26): the visual-fidelity upgrade — moving the avatar from "textured SMPL-X mannequin" toward a real likeness. New + tested: (1) `runtime::render_soft_aniso` — fully-differentiable **anisotropic EWA splatter** (projects full 3D covariance Σ=R·diag(s²)·Rᵀ through the perspective Jacobian to a 2D conic; oriented/elongated splats; **learnable rotation quaternions**), the fix for the blobby isotropic look; (2) `fit::fit_adaptive` + `fit::ssim` — full **adaptive 3DGS optimizer** (clone/split high-gradient Gaussians, prune transparent, opacity reset, L1+D-SSIM, per-view exposure); (3) `fit::fit_avatar`/`deform_avatar` — **animatable Gaussian avatar**: canonical Gaussians bound 1:1 to SMPL-X verts, LBS-skinned per frame, so multi-POSE casual video becomes multi-VIEW training for one avatar; (4) `ncg_cli avatar --frames dir/` — dir of frames → per-frame NLF pose+camera → trained avatar + fit-check/turntable. Validated on The Rock (16 usable frames from cinematic clips + photos): loss 0.21→0.099, **20.5 dB body-masked PSNR, 100% coverage**, training stable. Stabilization that mattered (build-blind): clamp scales [1e-3,0.05]; clamp splat power ≤0 (no exp overflow); **sanitize grads (NaN→0) before `clip_grad_norm_`** (its global norm otherwise lets one degenerate Gaussian poison every param → black avatar); `nan_to_num` render output. Tests: `test_render_aniso`, `test_fit_adaptive`, `test_avatar` all green. **Known limit: coverage is data-bound** — frontal-only sources give a sharp front, unconstrained back/sides; needs 360° capture, not a method change. See [[quality-improvement-leg]].

**END-TO-END WORKING on H100** (2026-06-25): casual photos+video → relightable, riggable, animated, Unity-ready avatar. Proven contributions (all benchmarked, `ncg_cli benchmark --seeds 5`): **C1** identifiability (albedo err 0.096→0.024 as photos↑), **C2** robustness (0.031 vs 0.096, 7× at 50% corruption), **C3** animate∘relight commutation (4.2e-7), **6.2×** better relighting than a NeRF/3DGS radiance baseline. Real-time **147 FPS** animate+relight runtime; **324 FPS** render. **Engine export**: rigged + animated (baked idle or extracted-mocap) glTF `.glb` → Unity/Unreal (`make_avatar.sh`). **Motion**: `tools/extract_motion.py` (video→SMPL-X motion) + `export/runtime --motion`. CLI: `fit fuse delight relight export benchmark runtime nerf`. **49 tests green** (2 design-skips). **Next contribution (C4, building): robust cross-action motion-style factorization** — motion identity, the analog of C1/C2 on the pose manifold.

**Method core + engine export GREEN on H100** (2026-06-24): full suite green (2 skips by design). The paper's technical core (`docs/method.md`) is implemented + benchmarked: `ncg-recon` multi-illumination **inverse-rendering solver** (SH L-step / per-vertex albedo A-step + robust consistency E-step + uncertainty) — `ncg_cli benchmark` on real SMPL-X geometry shows **C1** albedo err 0.096(N=1)→0.029(N=12) & relight err 0.227→0.052, **C2** robust ~flat vs naive 7× collapse under corruption, **relighting** 6.9% to a novel light. **Engine-ready export**: `mesh::write_glb`/`write_glb_skinned` + `ncg_cli export` write a valid (rigged, 55-joint) glTF for Unity/Unreal. `ncg_cli relight` shows the avatar relighting. SMPL-X now carries faces (`convert_smplx.py`). Per-subject 3DGS refinement (`fit --refine`) scales to 600px/4000 iters. Tests: inverse_render (C1/C2/relight), glb (static+skinned), appearance, init — all green.

**Phase 0 + Phase 1 GREEN on the H100** (2026-06-21): 27/29 tests pass, 2 skip by design (real-data SMPL-X + NLF golden — need vendored assets). Toolchain (CUDA 12.0 + libtorch 2.6 cu124 pre-cxx11-ABI, sm_90) validated; weight-port parity self-test green. Build via `scripts/ci.sh`.

**NeRF leg GREEN on H100** (2026-06-21, 4/4 nerf tests pass; total 33/36 with 2 by-design skips): `ncg-nerf` (`TinyNerf` MLP + differentiable `render_volume` + premultiplied `composite_over` hybrid + `fit_nerf_to_views`), `tests/nerf/test_nerf.cpp`, `ncg_cli nerf`. 3DGS + NeRF + hybrid now coexist. Single-view `fit_nerf_to_views` hardened against the dead-ReLU collapse (positive density-bias cold start + num_freqs 6→4 + lr 1e-3 with warm-up): fit descends `0.20 → 0.006` (~22 dB) on a body view. Also fixed a `ctest` LABELS bug (multi-label tests dropped every label after the first).

Phase 0 + Phase 1 scaffold + data-recording authored. Implemented: build system, `ncg-core`, `ncg-io` (image/safetensors/WeightMap/npy), parity harness (`test_golden_linear` self-test), `ncg-body` SMPL-X LBS forward, `ncg-recon`, `ncg-runtime` (camera + forward-splat CUDA kernel), `ncg-select` (sharpness), `ncg-record` (run dirs + metrics.jsonl + image dumps + timers + PSNR/SSIM/MAE, wired into the viewer so every stage records), `apps/ncg_viewer` + `apps/ncg_cli`, tests, `tools/` export/dump, `scripts/ci.sh`. **Stubs/skeletons:** `ncg::body::Nlf` (load/predict throw — first real port target, see `docs/parity.md`), `ncg-material`/`ncg-rig`/`ncg-mesh`. **`ncg::body::Nlf` works end-to-end on the H100** (2026-06-22): `torch::jit::load` the released `nlf_l_multi.torchscript`, call `detect_smpl_batched(frames_u8, model_name="smplx")` for a 55-joint pose matching SmplxModel, map → SMPL-X params → posed body → render. `ncg_cli fit --image me.jpg --weights models/nlf_l_multi.torchscript --smplx data/smplx_neutral.safetensors --out posed.png`. Two integration facts baked in: (1) pass `model_name="smplx"` (default "smpl" gives 24 joints SmplxModel can't use); (2) NLF's detector graph needs `torchvision::nms` — we **self-register our own `nms` C++ op** via `TORCH_LIBRARY(torchvision)` in `nlf.cpp` rather than dlopen torchvision's `_C.so` (which crashes a non-Python binary). `fit` canonicalizes NLF's camera-frame root orientation for an upright avatar (`--canonical 0` to keep it). Real SMPL-X vendored via `tools/convert_smplx.py`. Runtime needs `LD_LIBRARY_PATH=$TORCH/lib`; torchvision (Python) only needed for `tools/dump_nlf.py`, not the C++ runtime. Build/test module-by-module via `scripts/ci.sh`; see `docs/plan.md` progress log.

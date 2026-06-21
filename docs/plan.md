# NeuralCharGen — Master Plan (C++/CUDA)

**Goal:** a few casual, inconsistent personal photos (many low-quality) → a **hyperrealistic, rigged, relightable, full-body** avatar that drops into **Unity and Unreal**. Built **entirely in C++/CUDA**. Target: a new **SIGGRAPH** paper.

Grounded in two deep-research passes (see `research-findings.md` + memory). Supersedes the prior NeRFtrinsic+MTCM+FDNeRF+marching-cubes pipeline.

---

## 1. The contribution (three legs)

1. **Quality/uncertainty-aware fusion of inconsistent casual photos → one canonical avatar.** Beats closest prior art **PuzzleAvatar** (SIGGRAPH Asia 2024: album→mesh, but NOT rigged, NOT relightable, SDS/not-hyperreal, no bad-image handling).
2. **A real-time C++/CUDA animatable + relightable Gaussian-splat avatar runtime, embedded in Unreal & Unity.** Research confirms **no engine plugin does animate+relight splats** — this closes a real deployment gap and is the systems contribution that justifies the all-C++/CUDA build.
3. **Learned photo/view selection for avatar reconstruction** — confirmed underexplored (only robotics NBV exists). The principled successor to the old MTCM; the guided web-capture tool supplies the casual-vs-guided ablation statistics.

## 2. Representation decision — HYBRID, gap-filling

- **Mesh + PBR (SMPL-X-rigged):** guaranteed-shippable substrate; relights natively in any engine; collisions/shadows/LODs for free. The universal fallback.
- **Custom C++/CUDA splat-avatar runtime (hero path):** our own renderer doing LBS deformation + relighting in real time, embedded natively — NOT dependent on static-only plugins. This is leg #2 of the contribution.

## 3. Per-stage stack (port pretrained weights; train only novel modules)

| Stage | Method (port to C++/CUDA) | License (research OK) |
|---|---|---|
| Bad-upload selection | NR-IQA (TOPIQ/CLIP-IQA), CR-FIQA/DSL-FIQA (face), ArcFace ID clustering, **learned selector (novel)** | pyiqa/QualiCLIP NC; reimpl for shipping |
| Body anchor | **NLF** SMPL-X regressor | code MIT / weights NC |
| Geometry (casual/sparse) | **NoPo-Avatar** (animatable Gaussian, no pose) | check |
| Geometry (clothed mesh) | MultiGO++ / FRESA | research |
| Fusion across photos | **NOVEL** (leg #1) | ours |
| Game topology | Hunyuan3D Studio PolyGen | check |
| Intrinsic decomp / delight | **IDArb** (MIT) + **StableDelight** | IDArb MIT; weights TBD |
| Rigging | **UniRig** (or inherit SMPL-X rig) → retarget Unity Humanoid / Unreal | NC |
| Runtime renderer | **NOVEL** C++/CUDA splat+LBS+relight (leg #2) | ours |

## 4. C++/CUDA system architecture (module map)

- `ncg-core` — tensor/autograd substrate (**LibTorch-backed**) + custom CUDA kernel registry (recommended; from-scratch autograd optional for owned modules).
- `ncg-io` — image/dataset loading; **weight loader** (safetensors → internal binary) + forward-parity test harness.
- `ncg-select` — NR-IQA + FIQA + ArcFace clustering + learned selector. CUDA-accelerated.
- `ncg-body` — SMPL-X eval + NLF inference.
- `ncg-recon` — Gaussian avatar + **cross-photo fusion**; **custom 3DGS rasterizer (fwd+bwd CUDA)**.
- `ncg-material` — intrinsic decomposition / delighting → PBR (albedo/roughness/normal).
- `ncg-rig` — skinning weights + skeleton; retarget to Unity Humanoid / Unreal mannequin.
- `ncg-mesh` — mesh extraction + PBR bake (universal mesh path).
- `ncg-runtime` — **the deployable real-time avatar renderer**: LBS skinning + splat rasterization + relighting.
- `ncg-unreal` / `ncg-unity` — engine integration (native C++ / native plugin).

**Build:** CMake + CUDA Toolkit + LibTorch + CUTLASS (custom GEMMs) + Eigen.

## 5. Phased roadmap (baseline-first — prove the spine before optimizing)

> **Progress log** (newest first)
> - `2026-06-21` **NLF wired (port, not retrain)** — `ncg::body::Nlf` is now a `torch::jit::load` TorchScript loader, not a throwing stub: load the released `nlf_l_multi.torchscript`, run `detect_smpl_batched` on a uint8 RGB batch, pick a detection, map `pose/betas/trans` → `SmplxParams` (CPU, ready for `SmplxModel::forward`). `ncg_cli fit --image me.jpg --weights nlf_l_multi.torchscript --smplx smplx.safetensors --out posed.png` is the end-to-end image→posed-body→render path. `tools/dump_nlf.py` captures the real model's output-dict layout + a golden so the C++ parsing is confirmed (then `tests/golden/test_golden_nlf.cpp`). Decision locked: rewrite everything in C++/CUDA, **port** the commodity NLF weights (exact parity via the same TorchScript graph), train from scratch only the novel modules. ⏳ needs the downloaded NLF checkpoint + real SMPL-X to run.
> - `2026-06-21` **NeRF leg + hybrid path** (`ncg-nerf`) — ✅ GREEN on H100 (4/4 nerf tests pass): `TinyNerf` (positional-encoded MLP → density+RGB), differentiable `render_volume` (alpha-composited volume rendering matching the splat `RenderOutput`), `composite_over` (premultiplied-alpha front-to-back composite → Gaussian surface over NeRF volume = the hybrid renderer), `fit_nerf_to_views` (per-scene Adam overfit, no dataset — implicit-field counterpart to 3DGS fitting). Tests: ray unit-length/count, render shape/finiteness, composite opaque/transparent/half-cover, fit beats gray baseline. `ncg_cli nerf` demo (fit image → render; `--smplx` adds the hybrid composite dump). Answers the "keep NeRF / all-three" direction: 3DGS + real NeRF + hybrid now coexist. Single-view `fit_nerf_to_views` hardened against the dead-ReLU collapse (positive density-bias cold start + num_freqs 6→4 + lr 5e-3→1e-3 with 20-step warm-up): on a 256² body view the fit now descends `0.20 → 0.006` (~22 dB) instead of freezing at the background MSE.
> - `2026-06-19` Relighting + rigging baselines: `ncg-mesh::compute_vertex_normals`, `ncg-material::relight` (analytical Lambertian+ambient shader — real; learned IDArb decomposition stays gated), `ncg-rig::make_rigged` + `export_rigged` (OBJ + JSON skeleton/skinning; UniRig autorig + FBX/glTF gated). Tests for each. ⏳ awaiting first H100 build.
> - `2026-06-19` More phase coverage: **Phase 2 animation** (`SmplxOutput::vertex_transforms` + `recon::deform_gaussians` LBS deform, tested), **Phase 3 multi-view fit** (`fit_gaussians_to_views`, tested), **Phase 5 eval** (`runtime::orbit_trajectory` + `ncg_cli turntable`, tested), **Phase 4 geometry** (`ncg-mesh` real marching cubes + Gaussian density sampling + OBJ/PLY export, tested). Still gated: NLF weights/arch, learned relighting (IDArb), engine plugins, paper. ⏳ awaiting first H100 build.
> - `2026-06-19` Phase 2 start + SMPL-X unblock: `ncg-runtime::render_soft` (fully differentiable soft splatter, autograd reference for the CUDA rasterizer), `ncg-fit` (Adam optimization of a Gaussian cloud to a target image, per-iter loss/PSNR recording) + fit test (PSNR beats gray baseline), `ncg_cli fitimg` command. `tools/convert_smplx.py` (official `.npz`→safetensors) + `tools/make_dummy_smplx.py` (runnable UV-sphere body so the viewer runs without the licensed model). ⏳ awaiting first H100 build.
> - `2026-06-19` Data recording: `ncg-record` (timestamped run dirs, per-stage `metrics.jsonl`, image dumps, RAII timers, config snapshot) + `ncg-eval` metrics (PSNR/SSIM/MAE), wired into `ncg_viewer` so every stage records. Cross-cutting infra reused by all later phases. ⏳ awaiting first H100 build.
> - `2026-06-19` Phase 0 part 2 + Phase 1 scaffold (full): `ncg-io` (image/safetensors/WeightMap/npy), parity harness (`test_golden_linear` self-test, `tools/dump_golden.py`+`export_weights.py`, `docs/parity.md`), `ncg-body` (SMPL-X LBS forward + invariant tests; **NLF = stub port target**), `ncg-recon` (Gaussians on body), `ncg-runtime` (camera + forward-splat CUDA kernel + smoke test), `ncg-select` (sharpness), `ncg-material`/`ncg-rig`/`ncg-mesh` skeletons, `apps/ncg_viewer`+`ncg_cli`, `scripts/ci.sh`. ⏳ awaiting first H100 build; building module-by-module.
> - `2026-06-19` Phase 0 part 1: build system (CMake presets, cmake helpers, FetchContent deps), `ncg-core` (tensor/cuda/log helpers + saxpy kernel + vs-LibTorch test), `CLAUDE.md`, `docs/build.md`. ⏳ awaiting first H100 build to validate the toolchain.

- **Phase 0 — Foundations.** CMake/CUDA/LibTorch build; weight-port harness + numerical parity tests; tiny self-photo test set. *(part 1 done; part 2 = ncg-io + parity harness next)*
- **Phase 1 — Vertical slice (de-risk).** One good photo → SMPL-X fit → static Gaussian/mesh → render in our own C++/CUDA viewer. Ugly is fine; the win is an unbroken spine.
- **Phase 2 — Real-time runtime (leg #2).** Custom CUDA 3DGS rasterizer (fwd+bwd) + LBS deformation + relighting in a standalone viewer, then Unreal. The systems contribution.
- **Phase 3 — Fusion + selection (legs #1 + #3).** Multi-photo quality-aware fusion + learned selector; ablate vs single-photo / naive averaging / PuzzleAvatar.
- **Phase 4 — Relighting + universal mesh path.** Intrinsic decomposition → PBR; mesh+PBR export; Unity integration.
- **Phase 5 — Eval + paper.** Web-capture casual-vs-guided study, quantitative tables, user study, writing.

## 6. Weight-porting workflow (the timeline de-risker)
Reimplement architecture in C++/CUDA → load published weights (safetensors→binary) → **validate forward-pass numerical parity** against reference on fixed inputs → only then trust it. Train from scratch *only* the novel modules (fusion, selector, runtime).

## 7. Open / to-verify
- Topic 4: SMPL-X regressor head-to-head (NLF vs SMPLer-X/Multi-HMR/OSX) + weight licenses — only needed if commercializing; NLF is the working anchor.
- Volinga Pro (Oct 2025) reportedly relights 3DGS in Unreal — verify at build time; does not change the *animation* gap.
- Compute/VRAM/speed of IDArb/StableDelight/IQA models for batch over many uploads.

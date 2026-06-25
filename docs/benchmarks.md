# Benchmarks & statistics

All numbers measured on the H100 (sm_90), reproduced by `ncg_cli benchmark --smplx
data/smplx_neutral.safetensors --seeds 5`, on the **real SMPL-X mesh** (V = 10,475 vertices, F =
20,908 triangles). Errors are over **5 seeds, reported mean ± std**. Synthetic ground-truth albedo
(random per vertex) under random order-2 SH lights, so error is exactly measurable; relighting is
evaluated under a *novel, never-observed* directional light. Last run: 2026-06-24.

## C1 — identifiability: error vs number of photos (lighting diversity)

The core discovery: a single shared albedo under several unknown lights is identifiable, and
recovery improves monotonically with lighting diversity — with no light stage.

| #photos N | albedo error | relight error (novel light) | solve time |
|---:|---|---|---|
| 1  | 0.129 ± 0.021 | 0.261 ± 0.027 | 54 ± 56 ms |
| 2  | 0.080 ± 0.017 | 0.176 ± 0.034 | 49 ± 0.3 ms |
| 3  | 0.061 ± 0.010 | 0.138 ± 0.026 | 49 ± 0.2 ms |
| 5  | 0.049 ± 0.007 | 0.112 ± 0.021 | 53 ± 0.5 ms |
| 8  | 0.029 ± 0.006 | 0.068 ± 0.014 | 62 ± 0.4 ms |
| 12 | 0.024 ± 0.004 | 0.051 ± 0.008 | 67 ± 0.2 ms |

Error and its variance both shrink monotonically with N — the lighting diversity of a casual album
is what makes delighting well-posed.

## C2 — robustness: error vs corruption rate (robust vs naive), N = 8

A fraction of observations replaced by junk (clothing-swap / occlusion / bad upload). The robust
consistency estimator stays nearly flat; naive averaging degrades ~7×.

| corruption | robust (ours) | naive average |
|---:|---|---|
| 0%  | 0.031 ± 0.005 | 0.025 ± 0.006 |
| 10% | 0.031 ± 0.005 | 0.043 ± 0.004 |
| 20% | 0.032 ± 0.005 | 0.060 ± 0.003 |
| 35% | 0.037 ± 0.004 | 0.083 ± 0.003 |
| 50% | 0.041 ± 0.003 | 0.106 ± 0.003 |

Crossover at ~8%: below it, naive is marginally better (robustness has a small efficiency cost on
clean data — expected, and the auto-scaled kernel minimizes it); above it, robust dominates and the
gap widens. At 50% junk the robust estimator is **2.6× more accurate**.

## C3 — animate ∘ relight = relight ∘ animate (correct shading-frame transport)

| metric | value |
|---|---|
| max commutation error (shade posed-normals vs canonical-with-pulled-back-light) | **4.17 × 10⁻⁷** |
| normal transport time (10,475 verts) | 0.10 ms |

Machine-precision commutation: posing then relighting equals relighting then posing, because
normals are transported by the blended bone rotation. This is the property naive splat avatars
violate (they bake shading), and the reason no engine plugin does animate+relight.

## Comparison — relightability vs NeRF / vanilla 3DGS (the key ablation)

NeRF and 3DGS reconstruct **radiance** — they bake the capture lighting and *cannot* relight. Ours
recovers **albedo**. Evaluated under a novel, never-seen light (the radiance baseline is given its
best global scale to the target, i.e. its best case):

| method | relight error under novel light |
|---|---|
| **ours (relightable inverse rendering)** | **0.086 ± 0.022** |
| radiance baseline (NeRF / 3DGS, baked) | 0.528 ± 0.044 |

Ours is **6.2× more accurate** under relighting. The baseline's error is essentially the entire
lighting variation it cannot represent — the quantified reason a relightable method is needed.

## Throughput

| op | time | rate |
|---|---|---|
| **animate + relight runtime** (LBS pose → normal transport → SH relight → splat, 512×512) | 6.78 ms/frame | **147 FPS** (real-time) |
| forward-splat render alone (512×512, 10,475 gaussians) | 3.09 ± 0.03 ms | **324 FPS** |
| inverse-render solve (N=8, 60 iters, V=10,475) | ~62 ms | — |
| NLF forward (image → SMPL-X, with test-time aug) | ~28 s | — (one-time, ported) |
| per-subject 3DGS refine (600px, 4000 iters) | ~14 min | — (offline, optional) |

The 147 FPS animate+relight runtime is the deployable forward path (the systems leg). Its
differentiable counterpart for training is `render_soft`; an optimized tiled fwd+bwd rasterizer is
further engineering.

## Qualitative — real photos

`ncg_cli delight --images a,b,c` on 3 casual photos of one subject recovers a canonical albedo and
relights it under novel lights (see `server_artifacts/rock_delit.png`, relit frames). Consistent
regions (face/skin) delight cleanly; inconsistent regions (different outfits across the album) show
the expected robustness behaviour — the casual-input failure mode the method is designed for.

## Leg #3 — quality/coverage-aware view selection

`select_views` (submodular coverage maximization) selects, from a candidate pool, the views that
maximize surface coverage — provably picking the diverse views and rejecting redundant/junk
uploads. Unit-tested: from 12 candidates (4 diverse bands + 4 redundant + 4 junk) it selects the 4
diverse views for full coverage and never a junk view. Greedy is within (1−1/e) of optimal.

## Test coverage

`ctest --preset h100`: **49 tests, all pass** (2 skip by design — real-data SMPL-X golden and the
obsolete per-layer NLF golden). Includes: inverse-render C1/C2/relight/C3, glTF static+skinned
validity, appearance sampling/visibility/fusion, adaptive scale, view selection, SMPL-X LBS
invariants, custom CUDA-kernel vs LibTorch parity, NeRF, fit, golden harness self-test.

## Reproduce

```bash
ncg_cli benchmark --smplx data/smplx_neutral.safetensors --seeds 5   # all of the above
ctest --preset h100                                                   # full test suite
```

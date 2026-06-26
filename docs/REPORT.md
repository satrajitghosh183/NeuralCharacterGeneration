# NeuralCharGen — Playable Character Report

_Casual photos/video of a person → a hyperreal, rigged, animatable, physics-ready game character, in
two co-registered representations sharing one skeleton. Built entirely in C++/CUDA; no Python at
runtime. Last updated: 2026-06-26._

## 1. What you can do today

```bash
# Train an avatar from a folder of frames and export the playable character:
ncg_cli avatar --frames <frames_dir> --weights models/nlf_l_multi.torchscript \
  --smplx data/smplx_neutral.safetensors --max-frames 60 --res 384 --iters 4000 \
  --motion <locomotion.npy> --fps 24 --out-prefix rock_char
# -> rock_char.glb  (rigged + animated mesh)
#    rock_char.ply  (high-fidelity Gaussian splats) + rock_char.ply.skin (per-splat bone weights)
#    rock_char_fit0.png / _turn*.png (fit check + turntable)

# Play it (native Vulkan, on a machine with a display):
cmake -S apps/ncg_game -B build/game -DCMAKE_BUILD_TYPE=Release && cmake --build build/game
./build/game/ncg_game --glb rock_char.glb     # WASD walk, mouse orbit, Space jump
```

## 2. Architecture — one skeleton, two representations

The character is anchored on a **55-joint SMPL-X skeleton**. Two skins ride that skeleton:

| representation | file | strength | how it deforms |
|---|---|---|---|
| Rigged mesh | `*.glb` | universal, physics-ready, zero-plugin | standard glTF skinning |
| Gaussian splats | `*.ply` (+`.skin`) | photoreal render | `Σ_k w_k · boneMatrix[j_k]` (compute shader) |

Because both skin to the **same bones**, anything that drives the skeleton — an Animator clip,
retargeted mocap, or a **physics ragdoll** — deforms the mesh and the splats identically. The splat
skinning is the same math as the training-time `ncg::fit::deform_avatar`, and is **proven equal** to
it by `tests/fit/test_gs_skinning.cpp` (top-4 bone weights reproduce the full deform to < 1e-5).

## 3. Reconstruction pipeline (`ncg_cli avatar`)

1. **Per-frame NLF** → SMPL-X pose + a camera solved from the projected vertices (`solve_pinhole_camera`).
2. **Canonical Gaussians** seeded 1:1 on the SMPL-X vertices, colored from the first visible frame.
3. **Multi-pose training**: each frame is skinned to its pose (`deform_avatar`) and rendered through
   the **anisotropic EWA splatter** (`render_soft_aniso` — full 3D covariance → 2D conic, learnable
   rotations), supervised L1 + D-SSIM, per-frame exposure, body-masked. Multi-POSE casual video thus
   becomes multi-VIEW evidence for one avatar.
4. **Export** the skinned `.ply`/`.skin` + rigged `.glb`.

Stability hardening that mattered (all build-blind catches): clamp scales `[1e-3, 0.05]`; clamp the
splat power `≤ 0` (no `exp` overflow); **sanitize gradients (NaN→0) before `clip_grad_norm_`** — its
global norm otherwise lets one degenerate Gaussian poison every parameter (→ a black avatar);
`nan_to_num` on render output. Optional adaptive densification (`--densify 1`) clones/splits splats
with per-Gaussian vertex rebinding; it is bounded (the autograd soft renderer's memory scales with
N·H·W, so densify + high-res can OOM — use the tiled rasterizer for large N, future work).

## 4. Results (The Rock)

| run | frames | res | splats | body-masked PSNR |
|---|---|---|---|---|
| baseline (`rock_char`) | 16 | 320 | 10,475 | **20.8 dB** |
| high-fidelity (`rock_char_hi`) | ~30 | 384 | 10,475 | _<pending — updates on completion>_ |

Assets validated structurally: `.glb` = valid glTF v2 (55-joint skin, 65-frame animation); `.ply` =
standard 3DGS (10,475 finite splats, 17 properties); `.skin` = N records, joints 0–54, weights → 1.0.

**Honest limitation — coverage is data-bound.** The Rock inputs are frontal cinematic clips, so the
front resolves to a real likeness while the back/sides are unconstrained. This is data, not method: a
single slow **360° full-body phone video** (`ffmpeg -i orbit.mov -vf fps=8 frames/f_%04d.jpg`) feeds
the exact same command and fills all angles. No method change recovers geometry never photographed.

## 5. The game (`apps/ncg_game`)

Native **Vulkan** (vk-bootstrap + VMA + tinygltf + GLFW + glm), no engine. Loads the rigged `.glb`,
LBS-skins it in the vertex shader (joint matrices evaluated CPU-side per frame, same as
`deform_avatar`), renders it on a grid floor with an orbit camera, and walks it with WASD + a simple
gravity/jump. Runs on a display machine (M4 Pro via MoltenVK, or Linux+GPU) — not the headless H100.
Splat rendering in-engine is the follow-on (reuse the skinned buffers; see `unity/`).

## 6. Engine integration (`unity/`, `docs/engine_character.md`)

- `GaussianSplatSkin.compute` + `GaussianSplatRenderer.cs` + `SplatQuad.shader`: import the `.glb`
  (skeleton), drop in the component, assign the `.ply`/`.skin` — the splats skin to the rig every
  frame. Ragdoll Wizard on the main bones → the splats follow real physics.
- Unreal: `.glb` as Skeletal Mesh (PhysicsAsset ragdoll) + a GS plugin fed the skinned buffers.

## 7. Test coverage (new this leg)

`test_render_aniso` (anisotropic splatter + rotation gradients), `test_fit_adaptive` (densify +
D-SSIM reconstruction), `test_avatar` (LBS deform math), `test_gs_skinning` (engine skinning ==
`deform_avatar`). All green on the H100, alongside the prior 49.

## 8. Reproduce / files

- Pipeline: `ncg-fit/{fit_avatar,fit_adaptive}.cpp`, `ncg-runtime/renderer.cpp` (`render_soft_aniso`),
  `ncg-mesh/mesh_io.cpp` (`write_gaussian_ply`), `apps/ncg_cli/main.cpp` (`cmd_avatar`).
- Game: `apps/ncg_game/`. Engine: `unity/`, `docs/engine_character.md`.

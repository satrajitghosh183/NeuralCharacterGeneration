# Playable engine character — dual representation, one physics skeleton

`ncg_cli avatar` produces a game-ready character in **two co-registered representations that share a
single SMPL-X skeleton**, so you get Gaussian-splat fidelity *and* a portable, physics-driven body:

| file | what it is | role |
|---|---|---|
| `<prefix>.glb` | rigged (55-joint SMPL-X) skinned mesh, optional baked animation, per-vertex color | the **universal, physics-ready body** — imports into any engine, drives ragdoll/colliders/Animator |
| `<prefix>.ply` | standard 3DGS Gaussian splat (Inria convention) | the **high-fidelity render** — any Gaussian-splat plugin/viewer |
| `<prefix>.ply.skin` | per-splat top-4 bone indices + weights (binary: `int32 N`, then `N×(4×int32 + 4×float32)`) | binds each splat to the **same skeleton**, so the splats follow the physics |

The splats and the mesh are skinned to the **same 55 joints in the same order**, so whatever drives
the skeleton — an Animator clip, retargeted mocap, or a physics ragdoll — deforms both identically.

## Generate

```bash
ncg_cli avatar --frames <frames_dir> --weights models/nlf_l_multi.torchscript \
  --smplx data/smplx_neutral.safetensors --max-frames 60 --res 320 --iters 3000 \
  --motion <locomotion.npy> --fps 24 --out-prefix rock_char
# -> rock_char.glb (rigged+animated mesh), rock_char.ply (+ .skin), rock_char_fit0.png
```

Best input is a slow 360° full-body video (`ffmpeg -i orbit.mov -vf fps=8 frames/f_%04d.jpg`) so the
splats cover all sides. `--densify 1` adds Gaussians for finer face/hair (GS asset only; mesh keeps 1:1).

## Unity

**Physics body (works today, no plugins):**
1. Import `rock_char.glb` (Unity 2022+ has glTF import, or use *glTFast*). It comes in as a skinned
   mesh + a 55-bone skeleton; the baked clip plays via an Animator.
2. Configure the Avatar as **Humanoid** (map the SMPL-X bones: `pelvis→Hips`, `spine1..3→Spine/Chest`,
   `left_hip→LeftUpperLeg`, `left_knee→LeftLowerLeg`, `head→Head`, etc.). Now any humanoid clip retargets.
3. **Real physics:** run Unity's *Ragdoll Wizard* on the main bones (pelvis, spine, head, both
   upper/lower arms, both upper/lower legs). You get capsule colliders + CharacterJoints with mass —
   the body now collides, falls, and reacts to forces. For a player, add a `CharacterController`/
   `Rigidbody` for locomotion and switch to ragdoll for hit reactions (active ragdoll = blend
   animator pose with physics).

**Gaussian-splat fidelity (driven by the same skeleton):**
4. Import `rock_char.ply` with a Unity Gaussian-splat package (e.g. Aras-P *UnityGaussianSplatting*).
   It renders the splats statically out of the box.
5. To make the splats follow the rig/physics, attach a small skinning component that, each frame,
   reads the 55 bone matrices from the **same skeleton** (the glb's `SkinnedMeshRenderer.bones`) and,
   in a compute shader, transforms each splat by `Σ weight_k · boneMatrix[joint_k] · boneBind⁻¹`
   using `rock_char.ply.skin`. Splat position → skinned position; splat rotation (quaternion) →
   premultiplied by the blended bone rotation. This is the same `deform_avatar` math (`ncg-fit`).
6. Parent the splat renderer to the skeleton root so it inherits the physics body's world transform.

Result: the ragdoll/animator drives the SMPL-X skeleton; the mesh provides colliders + a fallback
render; the splats provide the photoreal look — all from one rig.

## Unreal

Same split: import `.glb` as a Skeletal Mesh (physics asset / `PhysicsAsset` for ragdoll on the main
bones), and the `.ply` via an Unreal Gaussian-splat plugin (e.g. *XScene/UEGaussianSplatting*), with a
skinning pass reading `.ply.skin` against the Skeletal Mesh's bone transforms.

## Notes / limits
- Coverage is data-bound: frontal-only inputs give sharp fronts and unconstrained backs. A 360° orbit
  fixes it (see above).
- SMPL-X includes hand/face joints; physics only needs the ~15 main body bones — skin uses all 55.
- The `.ply` is standard 3DGS, so it also opens in any web/desktop splat viewer for a quick look.

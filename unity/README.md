# Unity — skinned Gaussian-splat Rock (driven by the physics rig)

Renders the high-fidelity Gaussian-splat avatar (`rock_char.ply` + `rock_char.ply.skin`) and deforms
it every frame with the **same 55-bone SMPL-X skeleton** that drives the rigged mesh
(`rock_char.glb`). Animator clip *or* ragdoll → skeleton → both mesh and splats move identically.

The skinning algorithm is **proven correct**: `tests/fit/test_gs_skinning.cpp` shows the exact top-4
bone-weight math these files implement reproduces `ncg::fit::deform_avatar` to < 1e-5 (run it with
`ctest -R test_gs_skinning`).

## Files
| file | role |
|---|---|
| `GaussianSplatSkin.compute` | per-splat LBS: `Σ_k w_k · boneMatrix[j_k] · pos`, rotation composed too |
| `GaussianSplatRenderer.cs` | parses `.ply`+`.skin`, builds bone matrices from the rig each frame, dispatches the compute, draws the splats |
| `SplatQuad.shader` | compact camera-facing Gaussian-quad splatter (Built-in RP) |

## Setup (Unity 2022+, Built-in RP)
1. Import **rock_char.glb** (via *glTFast* or Unity's glTF importer) → a `SkinnedMeshRenderer` with the
   55-bone skeleton + bindposes. Add an `Animator` (the baked clip plays), or set up a ragdoll.
2. Add **rock_char.ply** and **rock_char.ply.skin** to the project, each renamed with a **`.bytes`**
   suffix so Unity imports them as `TextAsset` (e.g. `rock_char.ply.bytes`).
3. Create a Material from **SplatQuad.shader**.
4. New empty GameObject → add **GaussianSplatRenderer**; assign PlyFile, SkinFile, the imported
   `SkinnedMeshRenderer` (CharacterSMR), the compute shader, and the material. Press **Play**.

The splats now follow the animator/ragdoll. Hide the mesh's `MeshRenderer` (keep the skeleton) to see
splats only, or keep both (mesh as the physics/collider proxy, splats as the look).

## Physics
Run Unity's **Ragdoll Wizard** on the glb's main bones (pelvis, spine, head, upper/lower arms & legs).
The splats inherit the result because `GaussianSplatRenderer` reads `SkinnedMeshRenderer.bones` every
frame — the bones the ragdoll drives. See `docs/engine_character.md`.

## Fidelity notes
- The included shader is an **isotropic billboard** with a Gaussian falloff and no depth sort —
  enough to verify the skinned avatar moves correctly. For maximum quality (anisotropic conic +
  back-to-front sort), feed `GaussianSplatRenderer`'s skinned position/rotation `ComputeBuffer`s to a
  dedicated GS plugin (e.g. Aras-P *UnityGaussianSplatting*) instead of `OnRenderObject`'s draw call —
  the skinning stays identical.
- URP/HDRP: port `SplatQuad.shader` to the SRP (the C# + compute are pipeline-agnostic).

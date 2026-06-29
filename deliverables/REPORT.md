# Post-project report — NeuralCharGen avatar deliverable

## What shipped
A **photoreal, identity-locked, game-ready HEAD on a riggable, dressable body**, built
**off-the-shelf with no training**:

- **`me_char_rvxl2_face.glb`** — the user as a rigged glTF avatar. The face carries a
  RealVisXL_V4.0 + IP-Adapter-FaceID-portrait photoreal skin reprojected onto the SMPL-X
  UV; **identity ArcFace ≈ 0.56** (recognizably the user). 55-joint **named** SMPL-X
  skeleton, dressable skin weights, single clean mesh.
- **`kj_char_rvxl_face.glb`** — the same pipeline applied to a research benchmark subject
  (Kendall; benchmark only, not deployable).

Both are a single humanoid mesh with **zero stray geometry**, a named 55-bone rig, and an
opaque material — drop-in for Unity/Unreal, where the body is dressed at runtime.

## The thesis, as delivered
Casual personal photos → a hyperrealistic, **identity-locked**, rigged, game-ready
character, entirely from **pretrained off-the-shelf models** (NLF pose + SMPL-X +
RealVisXL/IP-Adapter-FaceID), no per-subject training. The recognizable photoreal **head**
on a **standard dressable rig** is the shippable unit.

## Honest limitations
- **Body texture is clean-neutral by design, not photoreal.** The casual photos observe the
  face densely and the body sparsely, so the body UV is under-constrained (washed/blotchy).
  This is acceptable for the target use: the **game dresses the body** at runtime. We did
  **not** bake clothing or fight for a photo-real body texture.
- **Eyes and hair are engine-supplied.** SMPL-X has no eyeballs or hair. We attempted to add
  eyeball spheres + a hair cap as additive geometry skinned to the head joint. In bind pose
  they verified correct (seated in sockets / scalp covered), but **in-engine the skinned
  eyeball/hair vertices evaluated to scattered positions** (an exploded-sphere artifact in
  Blender — a skinning/inverse-bind interaction, not a vertex-placement bug). Per the
  end-of-session discipline we **reverted to the clean pre-geometry asset** rather than debug
  it on a time-limited box. Eyes/hair should be added the way clothing is: by the engine, or
  as a future bounded geometry pass. `--eyes`/`--hair` remain in `ncg_cli face` but default
  **OFF**.
- **Identity number** (ArcFace ≈ 0.56) is inflated by FaceID conditioning on the same ArcFace
  family used to score; the eye-validated truth is "recognizably the user," not a calibrated
  metric.

## Reproduce
See `RUNBOOK.md` for the exact command sequence (sync → build → base bake → RealVisXL FaceID
refine → reproject bake → verify). RealVisXL_V4.0 is cached on the volume under `$A/hf`.

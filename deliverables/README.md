# Deliverables — shipped avatar assets

Engine-ready, rigged glTF avatars. These are the **clean, known-good** assets; the
experimental eyeball/hair geometry was reverted (it produced an in-engine skinning
artifact — see `REPORT.md`).

| File | Subject | What it is |
|------|---------|-----------|
| **`me_char_rvxl2_face.glb`** | user | hero asset — RealVisXL_V4.0 + IP-Adapter-FaceID photoreal face (identity ArcFace ~0.56, recognizable), 55-joint **named** SMPL-X rig, single clean body mesh, **zero stray geometry** |
| **`kj_char_rvxl_face.glb`** | Kendall (research benchmark only, not deployable) | same RealVisXL FaceID path, clean rigged body |

Both verified: single humanoid mesh (no exploded spheres), 55-joint rig, dressable
weights, opaque. **Eyes and hair are left to the engine** (like clothing) — see the
report for why.

## View
- macOS Finder: select the `.glb`, press **Space** (Quick Look renders + rotates it).
- Or drag into `gltf-viewer.donmccurdy.com`.

## Regenerate (clean head + rig, no experimental geometry)
```bash
ncg_cli face --frames data/me_clean_album \
  --smplx data/smplx_neutral.safetensors \
  --weights models/nlf_l_multi.torchscript \
  --out-prefix me_char --run me_char            # --eyes/--hair now default OFF
```
Full command sequence (incl. the RealVisXL skin bake) is in `RUNBOOK.md`.

## UPDATE — armature bug found; clean static meshes shipped
The "spheres" in Blender were the **broken armature**, not geometry: `write_glb_textured`
exports a malformed skeleton (leg/limb joints collapse to the origin), which Blender draws
as scattered bone shapes. The mesh itself is clean (1 mesh, 11313 verts).

**Ship these (no armature → no spheres):**
- `me_char_static.glb`, `kj_char_static.glb` — photoreal textured mesh, skeleton stripped.
  Re-rig in-engine (Mixamo auto-rig, or attach a correct SMPL-X armature).

TODO to restore rigging: fix the joint node translations in `write_glb_textured`
(`ncg-mesh/src/mesh_io.cpp`) — rest-pose joints collapse; verify the exported skeleton in
Blender, not just bind-pose renders.

## RESOLVED — armature fixed locally (working rig)
The skeleton bug is fixed. Root cause: the joints handed to `write_glb_textured` were
collapsed (most joints near origin), so the exported armature was garbage (Blender drew
it as scattered spheres). SMPL-X was unavailable (deleted), so the fix recovers each
joint from the glb's **own skin weights** — joint = weighted centroid of the vertices
shared between it and its parent (the articulation point) — then rewrites the node
translations + inverse-bind matrices. **Mesh/texture/weights are untouched** (bind pose
identical; FK == inverse-bind). Tool: `tools/fix_glb_rig.py <in.glb> <out.glb>`.

`me_char_rvxl2_face.glb` / `kj_char_rvxl_face.glb` now carry a correct, animatable
55-joint skeleton (verified: pelvis→hip→knee→ankle, spine→neck→head, shoulder→elbow→wrist,
finger chains — see `me_skeleton_FIXED.png`). The `*_static.glb` files remain as a no-rig
fallback.

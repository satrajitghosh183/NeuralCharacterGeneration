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

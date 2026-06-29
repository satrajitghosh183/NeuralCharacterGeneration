# Deliverables — preserved avatar assets

Engine-ready rigged + textured glTF avatars, committed here so the work survives
the ephemeral H100 (the box IP/instance changes on every unshelve). These are the
actual outputs of the `ncg_cli face` pipeline; regenerate from code + data with the
commands below.

| File | Subject | Notes |
|------|---------|-------|
| `me_char_rvxl2_face.glb` | user | best face bake — RealVisXL_V4.0 + IP-Adapter-FaceID-portrait reprojected onto the UV (identity ArcFace ~0.56, photoreal skin), 55-joint SMPL-X rig |
| `kj_char_rvxl_face.glb` | Kendall (research benchmark only, not deployable) | same RealVisXL FaceID bake path |
| `kj_ultimate_face.glb` | Kendall (research benchmark only) | KJ full combine — RealVisXL skin + eyes + hair + 55 named bones; all gates PASS |
| `me_eyes_hair_face.glb` | user | TASK 2 result — adds anatomically-placed **eyeballs** (skinned to head) and a dark **hair cap** (crown/back/sides, off the face) to the rig; all 55 bones **named**; analytic skin. All gates PASS. |
| **`me_ultimate_face.glb`** | user | **the full combine** — RealVisXL_V4.0 + IP-Adapter-FaceID-portrait_sdxl photoreal skin reprojected onto the UV **+ eyeballs + hair cap + 55 named bones**. All gates PASS (skeleton 55/55, dressable, opaque). The deployable hero asset. |

Each `.glb` is a 55-joint SMPL-X-rigged, UV-textured, opaque, dressable body
(engine adds clothing at runtime). Verified game-ready by `tools/glb_verify.py`.

## Regenerate
```bash
# face identity + texture bake (album -> rigged textured glb)
ncg_cli face --frames data/me_clean_album \
  --smplx data/smplx_neutral.safetensors \
  --weights models/nlf_l_multi.torchscript \
  --out-prefix me_char_rvxl2 --run me_char_rvxl2

# photoreal FaceID refine of the lit views, then reproject-bake onto the UV
python tools/faceid_views.py --lit-prefix runs/<run>/me_char_rvxl2 \
  --ref <reference.jpg> --out-dir /tmp/refine \
  --model SG161222/RealVisXL_V4.0 \
  --faceid-weight ip-adapter-faceid-portrait_sdxl.bin --strength 0.5
ncg_cli face ... --reproject-dir /tmp/refine   # bakes refined views -> UV
```
See `CLAUDE.md` (PHOTOREAL RECONSTRUCTION LEG) and the memory notes for the full
pipeline. Eyeball geometry (`--eyes 1`, default) adds anatomically-placed eyeballs
skinned to the head joint; hair geometry is the next additive leg.

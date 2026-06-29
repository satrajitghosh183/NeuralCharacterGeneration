# RUNBOOK — reproduce / continue the avatar pipeline

Everything below is what produced `deliverables/*.glb`. The H100 box IP changes on
every unshelve; current box: **`exouser@149.165.168.67`**, repo at
**`~/NeuralCharacterGeneration`**, assets/build on the volume
**`/media/volume/Prep_and_Voice_Training/ncg-assets`** (329 G free on `/dev/sdb`).
The off-limits voice/OLMo training also lives on this box — never `pkill`/`killall`;
kill specific PIDs; check `nvidia-smi` before GPU runs.

```bash
# 0. connect + env
ssh exouser@149.165.168.67
cd ~/NeuralCharacterGeneration
A=/media/volume/Prep_and_Voice_Training/ncg-assets

# 1. sync code + build the CLI (LD_LIBRARY_PATH/torch come from ~/.bashrc via `bash -lc`)
git fetch origin -q && git merge --ff-only origin/feat/phase0-1-foundation
cmake --build build/h100-release --target ncg_cli      # use `bash -lc "..."`

# 2. base bake: album -> rigged textured glb + EYES + HAIR + named bones + lit views
#    (--eyes/--hair default ON; tune --hair-q 0.85 --hair-front 0.42 --eye-back 0.018)
./build/h100-release/apps/ncg_cli face \
  --frames data/me_clean_album \
  --smplx data/smplx_neutral.safetensors \
  --weights models/nlf_l_multi.torchscript \
  --out-prefix me_base --run me_base --eyes 1 --hair 1
# -> runs/me_base_*/me_base_face.glb  and  me_base_face_lit_<az>.png  (az -40..40)

# 3. photoreal skin: RealVisXL + IP-Adapter-FaceID refine of the 5 lit views
source $A/face_venv/bin/activate
HF_HOME=$A/hf python tools/faceid_views.py \
  --lit-prefix runs/me_base_<TIMESTAMP>/me_base \
  --ref data/me_clean_album/p00.jpg \
  --out-dir /tmp/refine_me \
  --model SG161222/RealVisXL_V4.0 \
  --faceid-weight ip-adapter-faceid-portrait_sdxl.bin \
  --strength 0.5 --size 1024
# -> /tmp/refine_me/refine_<az>.png   (RealVisXL_V4.0 is cached under $A/hf, ~26 G)

# 4. ULTIMATE bake: reproject refined skin onto the UV + eyes + hair + named bones
./build/h100-release/apps/ncg_cli face \
  --frames data/me_clean_album \
  --smplx data/smplx_neutral.safetensors \
  --weights models/nlf_l_multi.torchscript \
  --out-prefix me_ultimate --run me_ultimate \
  --eyes 1 --hair 1 --reproject-dir /tmp/refine_me
# -> runs/me_ultimate_*/me_ultimate_face.glb   (the hero asset)

# 5. verify game-ready (skeleton named 55/55, dressable weights, opaque)
python tools/glb_verify.py runs/me_ultimate_<TIMESTAMP>/me_ultimate_face.glb

# 6. KJ: identical, with --frames data/kj and --ref data/kj/k000.jpg
```

## Verify renders (judge by eye — textured mesh can't render headless)
- `*_eyes_check_<az>.png` — eyeball placement (front/±25°)
- `*_hair_check_<az>.png` — hair coverage (front/back/side)
- `*_reproj_<az>.png` — baked skin (texel cloud)
- the final textured look: open the `.glb` in any glTF viewer

## Preserve results (nothing lost when the box dies)
```bash
# from the Mac repo: pull a glb off the box and commit it (no co-author trailers)
scp exouser@149.165.168.67:NeuralCharacterGeneration/runs/<RUN>/<name>_face.glb deliverables/
git add deliverables/ && git commit -m "deliverables: <name>" && git push origin feat/phase0-1-foundation
```

## Tunables (all CLI flags on `ncg_cli face`)
- Eyes: `--eye-x 0.032 --eye-up 0.030 --eye-back 0.018 --eye-r 0.012` (`--eyes 0` off)
- Hair: `--hair-q 0.85` (hairline height) `--hair-front 0.42` (face cutoff) `--hair-thick 0.018`
  `--hair-r/--hair-g/--hair-b` (colour, default near-black) (`--hair 0` off)

#!/usr/bin/env bash
# make_avatar.sh — upload casual photos, get back a rigged, textured glTF avatar for Unity/Unreal.
#
#   tools/make_avatar.sh photo1.jpg [photo2.heic ...] [--out my_avatar.glb]
#
# It converts your photos (HEIC→JPEG ok), uploads them to the H100, runs the NeuralCharGen
# pipeline (NLF → SMPL-X → appearance → rigged glTF), and downloads the .glb here. Drag the .glb
# into Unity (with the glTFast package) — see docs/unity.md.
set -euo pipefail

HOST="${NCG_HOST:-prepxl}"
REPO="NeuralCharacterGeneration"
TORCHLIB="\$HOME/.local/lib/python3.12/site-packages/torch/lib"
OUT="avatar.glb"
IMGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    *) IMGS+=("$1"); shift ;;
  esac
done
[ ${#IMGS[@]} -eq 0 ] && { echo "usage: make_avatar.sh img1 [img2 ...] [--out avatar.glb]"; exit 1; }

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
remotes=""
i=0
for f in "${IMGS[@]}"; do
  jpg="$tmp/up_$i.jpg"
  sips -s format jpeg -Z 1600 "$f" --out "$jpg" >/dev/null 2>&1 || cp "$f" "$jpg"
  scp -q "$jpg" "$HOST:$REPO/data/photos/up_$i.jpg"
  remotes="${remotes:+$remotes,}data/photos/up_$i.jpg"
  i=$((i + 1))
done
# 1 photo -> direct color sample; several -> fuse a clean albedo via the inverse-render solver.
if [ "$i" -gt 1 ]; then SRC="--images $remotes"; else SRC="--image $remotes"; fi
echo "uploaded $i image(s); building the avatar on the H100 (NLF ~30s/photo)…"

ssh "$HOST" "cd $REPO && export LD_LIBRARY_PATH=$TORCHLIB:\$LD_LIBRARY_PATH && \
  ./build/h100-release/apps/ncg_cli export \
    --smplx data/smplx_neutral.safetensors $SRC \
    --weights models/nlf_l_multi.torchscript --detection 0 --animate --out _make.glb"

scp -q "$HOST:$REPO/_make.glb" "$OUT"
echo "✅ done → $OUT"
echo "   Import into Unity with the glTFast package (Window → Package Manager → com.unity.cloud.gltfast),"
echo "   then drag $OUT into Assets/. See docs/unity.md."

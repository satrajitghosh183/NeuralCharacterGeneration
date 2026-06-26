#!/usr/bin/env bash
# best_shot.sh — extract the longest CONTINUOUS shot (no cuts) from a video and dump its frames.
#
# Why: avatar reconstruction needs *coherent* footage — one continuous shot of the person, where the
# camera/subject move to reveal angles. Montages, mixed clips and multi-person edits produce a
# blurry avatar because the shared appearance can't reconcile cuts/outfits/people. This tool mines
# the single most coherent segment out of any video (e.g. a downloaded interview) so the pipeline
# gets data it can actually fuse. Pair with several shots if you want more coverage, but keep each
# avatar fit to ONE coherent shot (or use --robust to reject the inconsistent minority).
#
# Usage:
#   tools/best_shot.sh <video.mp4> <out_frames_dir> [scene_threshold=0.3] [fps=6]
#   ncg_cli avatar --frames <out_frames_dir> --weights ... --smplx ... --out-prefix me
#
# Requires: ffmpeg, ffprobe, python3.
set -e
VID="${1:?usage: best_shot.sh <video> <out_dir> [thresh] [fps]}"
OUT="${2:?out_dir required}"
THRESH="${3:-0.3}"
FPS="${4:-6}"

DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VID")
echo "video: $VID  (${DUR}s)"

echo "detecting scene cuts (threshold $THRESH)..."
ffmpeg -hide_banner -i "$VID" -filter:v "select='gt(scene,$THRESH)',showinfo" -an -f null - 2>&1 \
  | grep -oE 'pts_time:[0-9.]+' | cut -d: -f2 > /tmp/_cuts.txt || true

read START LEN < <(python3 - "$DUR" <<'PY'
import sys
dur=float(sys.argv[1])
cuts=[0.0]+[float(x) for x in open('/tmp/_cuts.txt') if x.strip()]+[dur]
cuts=sorted(set(cuts))
best=max(((b-a,a) for a,b in zip(cuts,cuts[1:])), default=(dur,0.0))
print(f"{best[1]:.2f} {best[0]:.2f}")
PY
)
echo "longest continuous shot: ${LEN}s starting at ${START}s"

mkdir -p "$OUT"; rm -f "$OUT"/*.jpg
ffmpeg -hide_banner -loglevel error -ss "$START" -t "$LEN" -i "$VID" \
  -vf "fps=$FPS" -qscale:v 2 "$OUT/f_%04d.jpg"
echo "wrote $(ls "$OUT" | wc -l) frames -> $OUT"

# data/

Small test assets only. Large weights / golden dumps are git-LFS or fetched (see
`.gitattributes`, `docs/parity.md`) — do not commit raw model checkpoints.

```
data/
  golden/<model>/   manifest.json + input.npy + <stage>.npy + <model>.safetensors
                    (parity tests SKIP when a model's dir is absent)
  photos/           a few downscaled test photos for the Phase-1 slice
  weights/          ported model weights (.safetensors / .ncgw), via LFS
```

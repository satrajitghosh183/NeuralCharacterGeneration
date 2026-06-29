import sys

import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))


def emb(p):
    pil = Image.open(p).convert("RGB")
    w, h = pil.size
    pad = int(0.6 * max(w, h))
    c = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (128, 128, 128))
    c.paste(pil, (pad, pad))
    faces = app.get(np.asarray(c)[:, :, ::-1].copy())
    if not faces:
        return None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    e = f.normed_embedding
    return e / (np.linalg.norm(e) + 1e-9)


paths = sys.argv[1:]
embs = {p: emb(p) for p in paths}
for i in range(len(paths)):
    for j in range(i + 1, len(paths)):
        a, b = embs[paths[i]], embs[paths[j]]
        c = float(np.dot(a, b)) if (a is not None and b is not None) else None
        print(f"cos({paths[i].split('/')[-1]}, {paths[j].split('/')[-1]}) = {c}", flush=True)

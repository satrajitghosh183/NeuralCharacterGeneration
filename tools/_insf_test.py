import sys

import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
for p in sys.argv[1:]:
    img = Image.open(p).convert("RGB")
    bgr = np.asarray(img)[:, :, ::-1].copy()
    faces = app.get(bgr)
    print(p, "size", img.size, "-> faces:", len(faces), flush=True)
    if faces:
        e = faces[0].normed_embedding
        print("  emb", e.shape, "norm", float(np.linalg.norm(e)), "bbox", faces[0].bbox.tolist(), flush=True)

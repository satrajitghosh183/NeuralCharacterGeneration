# NeuralCharGen — system architecture

**One-line:** casual photos + casual video → a hyperreal, **relightable**, **riggable**, **personally-animated**, game-engine-ready 3D avatar — built entirely in C++/CUDA (no Python at runtime), on novel, benchmarked math.

**The unifying principle (the thesis):** *personal identity — how you look and how you move — is the invariant latent that explains inconsistent casual observations across nuisance diversity.* Lighting diversity makes **appearance** (albedo) identifiable; action diversity makes **motion style** identifiable. One principle, two manifolds, one avatar.

---

## 1. System diagram

```mermaid
flowchart TD
    subgraph IN[Casual input]
      P[A few casual photos<br/>different light/outfit/pose]
      V[Casual video clips<br/>walking, gesturing]
    end

    subgraph FE[Ported front-end - load weights, prove parity]
      NLF[NLF TorchScript<br/>image to SMPL-X + 2D/3D verts<br/>self-registered torchvision::nms]
      SMPLX[SMPL-X model<br/>LBS + blendshapes + faces]
    end
    P --> NLF
    V -->|per-frame, batched| NLF
    NLF --> SMPLX

    subgraph MAN[Shared substrate: the SMPL-X body manifold]
      M[(canonical surface M<br/>verts, normals, skeleton, skin)]
    end
    SMPLX --> M

    subgraph APP[Appearance identity - NOVEL]
      SC[sample_vertex_colors<br/>photo color at vertices2d]
      VIS[vertex_visibility<br/>z-buffer cull]
      IR["solve_inverse_render (C1+C2)<br/>per-photo SH light L-step<br/>+ per-vertex albedo A-step<br/>+ robust consistency E-step"]
      AB[(canonical albedo a<br/>+ per-vertex uncertainty)]
      SC --> VIS --> IR --> AB
    end
    M --> SC
    P --> SC

    subgraph MOT[Motion identity - NOVEL, in progress]
      EX[extract_motion<br/>video to SMPL-X pose seq]
      SF["style/content factorization<br/>pose = f(content_t, style)<br/>robust cross-action recovery"]
      ST[(personal motion style z)]
      EX --> SF --> ST
    end
    NLF --> EX

    subgraph REP[Representation]
      G[Gaussians on M<br/>adaptive scale + albedo<br/>bound to skeleton]
    end
    AB --> G
    M --> G

    subgraph RT[Real-time runtime - 147 FPS, the systems leg]
      AN[animate: LBS pose<br/>from style-conditioned motion]
      TR["transport_normals (C3)<br/>shading frame follows bones"]
      RL[relight: SH shading<br/>any light]
      SP[forward splat / soft renderer]
      AN --> TR --> RL --> SP
    end
    G --> AN
    ST --> AN
    AB --> RL

    subgraph OUT[Output]
      GLB[glTF .glb<br/>rigged + animated + colored]
      ENG[Unity / Unreal]
      IMG[relit / posed renders]
    end
    SP --> IMG
    G --> GLB --> ENG

    style APP fill:#1d3557,color:#fff
    style MOT fill:#3d1d57,color:#fff
    style RT fill:#1d573d,color:#fff
```

ASCII fallback (data flow):

```
casual photos ─┐                                    ┌─ albedo (relightable)  ──┐
               ├─► NLF ─► SMPL-X ─► body manifold ──┤                           ├─► Gaussians/mesh
casual video ──┘    (ported front-end)              └─ motion style (personal) ─┘        │
                                                                                          ▼
                                       animate(LBS) ─► transport normals(C3) ─► relight(SH) ─► splat
                                                                                          │  147 FPS
                                                                                          ▼
                                                                    glTF (.glb) ─► Unity / Unreal
```

---

## 2. Module map (`ncg-*`)

| Module | Role | Key pieces |
|---|---|---|
| `ncg-core` | tensor/CUDA/log helpers, custom-kernel pattern | error/cuda checks, saxpy golden |
| `ncg-io` | image, **safetensors**, WeightMap, **npy** | zero-copy mmap, parity harness |
| `ncg-body` | image→SMPL-X | **NLF** (TorchScript loader + self-registered `nms`), SMPL-X LBS forward, faces |
| `ncg-recon` | **the novel core** | appearance (sample/visibility/fusion), adaptive scale, **inverse_render (C1/C2 + SH + relight + C3 transport)** |
| `ncg-runtime` | renderers + camera | forward-splat CUDA kernel, differentiable `render_soft`, `solve_pinhole_camera` |
| `ncg-fit` | per-subject 3DGS | image/multi-view fit, **`refine_gaussians_to_image`** (camera-solved, masked) |
| `ncg-mesh` | mesh + export | marching cubes, normals, **glTF write (static / skinned / animated)** |
| `ncg-select` | bad-upload handling | sharpness + **`select_views`** (submodular coverage, leg #3) |
| `ncg-nerf` | NeRF leg + hybrid | TinyNerf, differentiable volume render, `composite_over` |
| `ncg-material`/`ncg-rig` | relight shader / rig export | analytical relight, OBJ+rig JSON |
| `ncg-record` | experiment recording | run dirs, metrics.jsonl, image dumps, timers, PSNR/SSIM/MAE |
| `apps/ncg_cli` | driver | `fit fuse delight relight export benchmark runtime nerf render turntable select` |
| `tools/` | DEV-only Python | `convert_smplx`, `dump_nlf`, **`extract_motion`**, `make_avatar.sh` |

---

## 3. The three+1 contributions (where the novelty lives)

| | Claim | Validated |
|---|---|---|
| **C1** | casual multi-illumination ⇒ albedo identifiable; diversity helps | benchmark: 0.096→0.024 |
| **C2** | robust estimator rejects inconsistent/junk observations | 0.031 vs 0.096 (7× at 50%) |
| **C3** | animate∘relight = relight∘animate (normal transport) | error 4.2e-7 |
| **comparison** | relighting vs NeRF/3DGS (they bake light) | 6.2× better |
| **C4 (new)** | cross-action ⇒ **motion style** identifiable + robust from casual video | *building now* |

Math + derivations: `docs/method.md`. Numbers + stats: `docs/benchmarks.md`.

---

## 4. End-to-end pipeline (commands)

```bash
# photos -> rigged, textured, animated avatar (Unity-ready), one command:
tools/make_avatar.sh front.jpg side.jpg --out me.glb

# pieces:
ncg_cli fit       --image me.jpg --weights nlf --smplx smplx --refine   # 3DGS refine
ncg_cli fuse      --images a,b,c ...                                     # cross-photo fusion
ncg_cli delight   --images a,b,c ...                                     # recover albedo + relight
ncg_cli relight   --smplx smplx                                         # orbiting-light relight
ncg_cli export    --image me.jpg --weights nlf --smplx smplx --animate  # rigged+idle glTF
ncg_cli runtime   --smplx smplx [--motion m.npy]                        # 147 FPS animate+relight
ncg_cli benchmark --smplx smplx --seeds 5                               # paper figures
tools/extract_motion.py --frames dir --out m.npy                        # video -> SMPL-X motion
```

---

## 5. Hardware / dev loop

Author on Mac (no CUDA) → push → build/run on **Jetstream2 H100 (sm_90)** → results back.
LibTorch = the torch 2.6 wheel (pre-cxx11-ABI). Full test suite: **49 green, 2 design-skips**.
Build-blind discipline: every kernel has a vs-LibTorch test; every claim has a benchmark.

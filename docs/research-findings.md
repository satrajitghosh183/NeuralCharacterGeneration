# NeuralCharGen — SOTA Research Findings & Locked Stack (June 2026)

Goal: casual, inconsistent personal photos (many low-quality) → **hyperrealistic, rigged, relightable, full-body** avatar that drops into Unity **and** Unreal. Target: a new SIGGRAPH paper.

Source: deep-research run 2026-06-19 (23 sources, 108 claims, 25 adversarially verified, 22 confirmed) + PuzzleAvatar manual verification.

## Headline verdict
Every pipeline **stage** has a strong building block, but **no published method does the whole thing end-to-end from casual input** — that intersection is the novelty. Confirmed by HumanOLAT (ICCV 2025): full-body relighting+NVS "remains significantly limited." Every hyperreal/relightable full-body method needs a **light stage or monocular video**; every casual/few-image method is **neither relightable nor verified on inconsistent inputs**.

## Per-stage stack (recommended / fallback / license)

| Stage | Recommended | Fallback | License note |
|---|---|---|---|
| Bad-upload filtering & view selection | NR-IQA (TOPIQ/MANIQA/CLIP-IQA via `pyiqa`) + face quality (CR-FIQA) + ArcFace identity clustering + CLIP outfit clustering | Laplacian-variance blur + heuristics | all permissive — **under-evidenced, needs follow-up** |
| Body pose+shape anchor | **NLF** (NeurIPS 2024) — SOTA SMPL-X regression | SMPLer-X, Multi-HMR, OSX | code MIT; **weights noncommercial-research** (fine for a paper) |
| Geometry — sparse/casual | **NoPo-Avatar** (NeurIPS 2025) — animatable Gaussian T-pose, **no pose needed**, 1–3 imgs | FRESA (few-img skinned mesh) | check repo licenses |
| Geometry — single-img clothed mesh | **MultiGO++** (2026) — complete textured clothed mesh | InstantMesh (general) | research |
| Geometry — casual album, multi-outfit | **PuzzleAvatar** (SIGGRAPH Asia 2024) ← **prior-art baseline** | — | noncommercial-research |
| Game-ready topology | **Hunyuan3D Studio PolyGen** — low-poly, deformation-aware | mesh decimation + remesh | check |
| Hyperreal appearance + relighting | **Gaussians bound to template** (TaoAvatar, 90 FPS on Vision Pro) + **PBR/Disney-BRDF decomposition** (RnD-Avatar) | mesh + baked PBR | research |
| Relighting upper-bound refs | Relightable Full-Body Gaussian Codec Avatars (SIGGRAPH 2025), BecomingLit (NeurIPS 2025) | — | light-stage only — **not usable from casual**, reference targets |
| Auto-rigging | **UniRig** (SIGGRAPH 2025) mesh→skeleton+skinning FBX | inherit SMPL-X rig directly | noncommercial-research |
| Engine deploy | Unity: aras-p UnityGaussianSplatting; Unreal: DazaiStudio SplatRenderer; **mesh+PBR native to both** | — | **plugin maturity under-evidenced** |

## Mesh-vs-splat decision → HYBRID
- **Splat (3DGS)** wins hyperrealism (skin/hair/cloth) but in-engine animation+relighting is immature and plugin-dependent.
- **Mesh+PBR+SMPL-X rig** is the only path that *guarantees* "drops into any engine, relightable, animatable."
- **Decision:** dual representation — a rigged mesh+PBR substrate (universal, relightable, droppable) **plus** a Gaussian detail layer bound to the rigged template (TaoAvatar-style) for hyperrealism where the engine supports splats. The dual-rep bridge is itself part of the contribution.

## Novelty angles (positioned against prior art)
**Closest prior art = PuzzleAvatar** (album → textured clothed mesh, multi-outfit). Its gaps: **not rigged, not relightable, SDS-diffusion (not photoreal/hyperreal), no bad-image selection, slow per-subject optimization.**

- **A — Quality/identity-aware fusion of inconsistent casual photos → single canonical avatar.** Delta over PuzzleAvatar: hyperreal + rigged + relightable + explicit selection/uncertainty over bad uploads. (FRESA's casual-photo & cross-photo-fusion claims were *refuted* in verification — the fusion problem is genuinely open.)
- **B — The full intersection {casual inconsistent photos} × {hyperreal} × {full-body rigged} × {relightable, game-ready} has no published solution** (SIGGRAPH'25 codec avatars need a 512-cam/1024-light stage).
- **C — Informativeness/uncertainty-aware aggregation** — the principled successor to the old MTCM: learn which of many bad uploads contribute reliable identity/geometry signal. Maps directly to "people upload bad images and you have to select." Guided web-capture provides the controlled ablation/statistics.

## Under-evidenced — needs a focused 2nd research pass
1. 3DGS Unity/Unreal plugin maturity: can splat avatars be **animated + relit in-engine**, or is mesh-bake required?
2. NR-IQA / face-quality / learned view-selection standards for filtering bad uploads.
3. Intrinsic decomposition tooling (IDArb etc.) for the relighting stage.
4. SMPL-X regressor head-to-head robustness (NLF vs SMPLer-X vs Multi-HMR vs OSX) + commercial license terms.
5. Deeper album-avatar prior-art sweep (PuzzleAvatar, PSHuman, others) to firmly establish the gap.

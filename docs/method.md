# Method — the technical core (paper draft)

> Status: research formulation, 2026-06-22. This is the *method* the paper is about — not the
> pipeline. The pipeline (`docs/plan.md`) is the vehicle and the evidence; this document is the
> contribution. Honest about assumptions and what still needs proof/experiment.

## 0. One-sentence thesis (the discovery)

**A few casual phone photos already contain a multi-illumination dataset.** Each photo captures
the same person under *different, unknown* lighting. Treated naively this inconsistency is noise;
treated correctly it is the **signal that makes a relightable avatar identifiable without a light
stage** — because a single shared albedo must explain every photo under its own lighting, which
breaks the albedo–shading ambiguity that defeats single-image methods. We recover a relightable,
animatable canonical avatar from inconsistent casual photos by a robust, consistency-gated,
multi-illumination inverse-rendering estimator on the body manifold, and we render it with a
representation whose shading transport makes **animation and relighting commute**.

This reframes "casual album → avatar" from an *integration* problem into an *inverse problem* with
a non-obvious identifiability result. That is the SIGGRAPH-shaped claim.

---

## 1. Why it is hard (taxonomy of inconsistency)

Casual photos differ in ways consistent-scene multi-view methods assume away:

| Nuisance | Per-photo? | Handled by |
|---|---|---|
| Lighting (intensity, direction, color) | yes, unknown | per-photo SH lighting $L_i$ — **and exploited** (§2) |
| Camera + global pose | yes | NLF + residual $\delta_i$ |
| Articulated pose | yes | SMPL-X manifold pooling (§3) |
| Clothing change | per-region | structured consistency field $w$ (§4) |
| Occlusion / cropping | per-region | visibility $\nu$ + $w$ |
| Identity drift / junk uploads | whole photo | $w$ + selection (§7) |

Prior work either assumes consistency (multi-view stereo/NeRF), needs a light stage (relightable
full-body), or ignores relighting/robustness (PuzzleAvatar, album→mesh). None recovers a
*relightable* avatar from *inconsistent* casual input. That gap is the target.

---

## 2. Generative model

Canonical body manifold $\mathcal{M}$ = the SMPL-X surface (vertices $v=1..V$, normals
$n_v^{\text{can}}$). The unknowns of interest live on $\mathcal{M}$ and are therefore **shared
across all photos and poses** — this is what makes cross-pose pooling well-posed (color keyed by
vertex identity, already exploited in `recon::sample_vertex_colors`).

- **Albedo field** $a:\mathcal{M}\to\mathbb{R}^3$ (per-vertex Lambertian albedo $a_v$; extends to a
  spatially-varying BRDF, §8).
- **Per-photo lighting** $L_i$ — order-2 spherical harmonics, $\ell_i^c\in\mathbb{R}^9$ per color
  channel $c$ (Ramamoorthi–Hanrahan: Lambertian reflectance is captured to ~99% by 9 SH terms).
- **Per-photo pose/camera** from NLF, with a small learned residual $\delta_i$.

Posed world normal at vertex $v$ in photo $i$: $n_{iv}=R_{iv}\,n_v^{\text{can}}$, where $R_{iv}$ is
the LBS-induced rotation (§9). Lambertian SH shading gives irradiance
$E_i^c(n)=\ell_i^{c\top} b(n)$ with $b(n)\in\mathbb{R}^9$ the SH basis (incl. the convolution
factors $\hat A_\ell$). Predicted color:
$$\hat I_{iv}^c = a_v^c\,\big(\ell_i^{c\top} b(n_{iv})\big).$$
Observation $I_{iv}$ = photo $i$ bilinearly sampled at the projection of $v$, present iff visible
($\nu_{iv}=1$) and consistent ($w_{iv}\!\to\!1$).

---

## 3. The objective (robust, consistency-gated, manifold-regularized)

Negative log-posterior over albedo, per-photo lighting, pose residuals, and a **latent
consistency field** $w_{iv}\in[0,1]$:

$$
E \;=\; \sum_{i,v}\nu_{iv}\,w_{iv}\,\frac{\lVert I_{iv}-a_v\odot(\boldsymbol\ell_i^\top b(n_{iv}))\rVert^2}{2\sigma^2}
\;+\;\sum_{i,v}\psi(w_{iv})
\;+\;\lambda_a\,\mathcal R_a(a)\;+\;\lambda_L\sum_i\mathcal R_L(\ell_i)\;+\;\lambda_\delta\sum_i\lVert\delta_i\rVert^2 .
$$

- $\psi(w)$ is the **half-quadratic dual** of a robust kernel $\rho$ (Geman–McClure / Welsch):
  $\rho(r)=\min_{w\ge0} w\,r^2+\psi(w)$. Minimizing over $w$ *is* robust estimation, but we go
  beyond plain IRLS:
- **Structured consistency prior.** $w$ is not i.i.d. — a clothing change is a *spatially coherent*
  low-$w$ region on $\mathcal{M}$ and *correlated across* the photos that see that garment. We add
  a graph–total-variation term on $\mathcal{M}$ and a low-rank coupling across photos:
  $\Psi(w)=\beta_{\text{TV}}\sum_{(u,v)\in\mathcal E}\lvert w_{iu}-w_{iv}\rvert + \beta_{\text{rank}}\lVert W\rVert_*$.
  This lets the estimator *distinguish* "occlusion/garment" (structured) from "noise" (isolated) —
  the part that is genuinely new versus a robust kernel.
- $\mathcal R_a$ = manifold-Laplacian smoothness on albedo (+ optional chroma/gray-world prior to
  fix the global gain, §5). $\mathcal R_L$ = SH smoothness/non-negative-irradiance prior.

---

## 4. Identifiability (why casual ⇒ relightable; the crux)

**Single image.** $I=a\odot E(n;L)$ has the classic albedo–shading ambiguity: unknowns $a$ ($3V$)
and $L$ ($27$) vs. observations ($3|\text{vis}|$); the $a\!\leftrightarrow\!L$ split is
unidentifiable without strong priors. This is why single-image intrinsic decomposition is
ill-posed and why one photo can't be relit faithfully.

**$N\!\ge\!2$ images, shared albedo, independent lighting.** Unknowns $3V+27N$; observations
$3\sum_i|\text{vis}_i|$. Any vertex seen in $\ge2$ photos contributes $\ge2$ equations for the
*same* $a_v$ under *different* SH operators $E_i$. Stacking over photos, the shared-albedo
constraint over-determines the per-image split and collapses the ambiguity to a **single global
per-channel gain** $a\!\to\!\kappa a,\ L\!\to\!\kappa^{-1}L$, removable by one scalar prior
(gray-world / known max-albedo).

> **Proposition (informal).** If $\ge2$ photos observe each region under lighting that is not a
> common scaling, and coverage is connected on $\mathcal M$, then $(a,\{L_i\})$ is identifiable up
> to one global per-channel scale. **Discovery:** the lighting *diversity* of a casual album is
> precisely the condition that makes full-body delighting well-posed with no light stage.

> **Empirical support (2026-06-22, `tests/recon/test_inverse_render.cpp`, green on H100):** with
> the closed-form solver (§5), scaled albedo error is **0.068 at N=1 vs 0.024 at N=5** diverse
> lights — recovery is good under multi-illumination and *strictly improves with lighting
> diversity*, exactly as C1 predicts. (Idealized: synthetic Lambertian, no noise/outliers yet;
> the noise/outlier + error-vs-N sweep is the next experiment.)

Honest caveats: this is a DOF-counting + cross-illumination argument, not yet a theorem under a
formal genericity assumption; degenerate albums (all photos same light) collapse to the
single-image case — detectable from the conditioning of the L-step normal equations and reportable
as an uncertainty (§6). A clean proof + the exact genericity condition is a paper sub-result to
nail.

---

## 5. Solver (EM / block-coordinate; each block closed-form & CUDA-parallel)

Alternate to convergence (warm-started from the naive photo-color fuse we already have):

1. **E-step — consistency $w$.** Given residuals $r_{iv}$, $w_{iv}\!=\!\arg\min_w w r_{iv}^2+\psi(w)$
   (closed form per kernel; e.g. Welsch $w=\exp(-r^2/2\sigma^2)$), then apply the structured prior
   $\Psi$ by a few proximal/graph-cut sweeps on $\mathcal M$. Massively parallel.
2. **L-step — lighting (closed form).** Fix $a,w$. Per photo $i$, channel $c$, the model is linear
   in $\ell$: a $9\times9$ weighted normal-equation solve
   $$\ell_i^c=\Big(\textstyle\sum_v w_{iv}\nu_{iv}(a_v^c)^2 b_{iv}b_{iv}^\top\Big)^{-1}\sum_v w_{iv}\nu_{iv}a_v^c I_{iv}^c b_{iv}.$$
   Its conditioning = the identifiability monitor of §4.
3. **A-step — albedo (closed form, per vertex ⇒ embarrassingly parallel).** Fix $\ell,w$. Per
   vertex $v$, channel $c$:
   $$a_v^c=\frac{\sum_i w_{iv}\nu_{iv}\,(\ell_i^{c\top} b_{iv})\,I_{iv}^c}{\sum_i w_{iv}\nu_{iv}\,(\ell_i^{c\top} b_{iv})^2}.$$
   Then one manifold-Laplacian smoothing solve (a few Jacobi/CG iterations) applies $\mathcal R_a$.
   This is a custom CUDA kernel (one thread per vertex) — fits the all-C++/CUDA thesis.
4. **δ-step (optional) — pose/camera residual.** Gauss–Newton on the photometric + reprojection
   residual; small.

Convergence: each block is a (weighted) least-squares or a proximal step on a bounded-below energy
⇒ monotone descent. Cost is $O(\sum_i|\text{vis}_i|)$ per sweep; real-time-feasible for $V\!\sim\!10^4$,
$N\!\sim\!10$.

---

## 6. Uncertainty (a first-class output, not an afterthought)

The A-step denominator $\Lambda_v^c=\sum_i w_{iv}\nu_{iv}(\ell_i^{c\top}b_{iv})^2$ is the
(Gauss–Newton) **precision** of $a_v^c$ ⇒ posterior variance $\approx\sigma^2/\Lambda_v^c$. This
gives, for free:
- a principled **per-vertex confidence** (replaces the heuristic visibility weight in
  `fuse_vertex_colors`),
- **uncertainty-driven completion**: low-precision vertices (never well-lit / never seen) are
  filled by a learned prior or bilateral symmetry, *and flagged*,
- a **capture-guidance signal**: tell the user which direction/lighting to shoot next (active
  acquisition — D-optimal next view), which is the rigorous version of "guided capture."

---

## 7. Selection as experimental design

Bad-upload handling becomes: choose a subset $S$ of photos maximizing information about $a$,
$\max_S \log\det\big(\sum_{i\in S}\mathcal I_i\big)-\gamma|S|$ where $\mathcal I_i$ is photo $i$'s
Fisher information for the albedo (from §6). Junk photos contribute ~no information and are dropped;
this is a submodular objective (greedy gives a $1-1/e$ guarantee) — not a sharpness heuristic.

---

## 8. Representation: animatable **and** relightable manifold-bound Gaussians

The systems gap (no engine animates *and* relights splats) is a transport-math gap. Bind each
Gaussian $g$ to a surface triangle: canonical mean $\tilde\mu_g$, covariance $\tilde\Sigma_g$,
albedo $a_g$ (from §5), canonical shading frame $n_g^{\text{can}}$ (+ tangent).

Under LBS pose $\theta$ with skin weights $W_{gb}$ and bone transforms $T_b(\theta)$:
- geometry: $\mu_g(\theta)=\sum_b W_{gb}T_b(\theta)\,\tilde\mu_g$;
- the **shading frame must transport**: let $\overline R_g(\theta)=\mathrm{polar}\big(\sum_b W_{gb}R_b(\theta)\big)$
  (rotation of the blended linear part). Then world normal $n_g(\theta)=\overline R_g(\theta)n_g^{\text{can}}$,
  covariance $\Sigma_g(\theta)=\overline R_g\tilde\Sigma_g\overline R_g^\top$;
- shade at render time with the *recovered* albedo and *target* light $L^\star$:
  $$c_g(\theta,L^\star)=a_g\odot E\big(\overline R_g(\theta)\,n_g^{\text{can}};\,L^\star\big).$$

**Property (animation–relighting commute).** Because color is computed from the transported normal
at render time rather than baked, $\textsf{relight}(L^\star)\!\circ\!\textsf{animate}(\theta)=
\textsf{animate}(\theta)\!\circ\!\textsf{relight}(L^\star)$. Naive splat avatars bake shading and
violate this — which is *why* current plugins can't animate+relight. The corrected transport (and
its differentiable form, so $a_g$ is learned through it in §5) is the representational
contribution.

---

## 9. Novelty vs. prior art

| Capability | PuzzleAvatar (SA'24) | light-stage relightable | single-img intrinsic | **Ours** |
|---|---|---|---|---|
| casual / few-photo input | ✅ | ❌ | ✅ | ✅ |
| inconsistency-robust (clothes/junk) | ❌ | n/a | ❌ | ✅ (§3 structured $w$) |
| **relightable, no light stage** | ❌ | ✅ (needs stage) | ⚠ ill-posed | ✅ (§2,4 discovery) |
| rigged / animatable | ❌ | varies | ❌ | ✅ (§8) |
| animate **and** relight together | ❌ | ❌ | ❌ | ✅ (§8 transport) |
| uncertainty / capture guidance | ❌ | ❌ | ❌ | ✅ (§6) |
| real-time C++/CUDA, engine-native | ❌ | ❌ | ❌ | ✅ |

The contributions claimed in the paper: **(C1)** the multi-illumination identifiability result for
casual full-body delighting (§4); **(C2)** the robust, structured-consistency manifold
inverse-rendering estimator + parallel solver (§3,5,6); **(C3)** the commuting animate+relight
Gaussian transport (§8). The end-to-end system is the demonstration, not the claim.

---

## 10. Experiments that prove the math earns its keep

- **Synthetic, ground-truth (the proof).** Render a SMPL-X subject with known albedo under $N$
  random SH lights, varied poses, simulated clothing swaps + junk frames. Recover and measure:
  albedo error vs. $N$ and vs. lighting diversity (validates §4), relighting error under *novel*
  light, robustness vs. % corrupted frames.
- **Ablations (the baselines to beat — what we built is row 1).**
  1. naive visibility-weighted color average (current `fuse_vertex_colors`),
  2. + per-photo SH delighting (no robustness),
  3. + robust kernel (plain IRLS),
  4. + structured consistency field (full **C2**),
  vs. single-photo, vs. PuzzleAvatar (geometry/appearance), vs. a light-stage upper bound.
- **Real casual albums** + a guided-capture user study (casual vs. guided, §6 guidance).
- **Runtime**: animate+relight FPS in Unreal/Unity; correctness of the commutation (§8) vs. a
  bake-per-pose baseline.

Metrics: albedo PSNR/Δ, relit-PSNR/SSIM/LPIPS under held-out light, identity (ArcFace), reprojection.

---

## 11. What is genuinely new vs. assumed (honesty)

- **New:** C1 (identifiability + "inconsistency-as-signal" framing), C2 (structured-consistency
  manifold inverse rendering + uncertainty), C3 (commuting animate+relight transport).
- **Ported / assumed (not claimed):** NLF (image→SMPL-X), SMPL-X, SH Lambertian shading,
  3DGS rasterization, half-quadratic/IRLS robust estimation (we *extend* it with structure).
- **Risks / open:** (a) formalizing §4 with a genericity condition; (b) Lambertian assumption —
  specular hair/skin/clothing needs the BRDF extension (§8) and may need a learned residual; (c)
  geometry comes from SMPL-X (no clothing geometry yet — PuzzleAvatar has clothed mesh); (d)
  global-scale gauge needs a stable prior. Each is a stated limitation, not hidden.

---

## 12. Build path (replace the placeholder with the method)

1. SH basis + Lambertian shading eval (CPU + CUDA), unit-tested vs. analytic.
2. **L-step** (per-photo 9×9 SH solve) and **A-step** (per-vertex parallel albedo solve) — replace
   `recon::fuse_vertex_colors`; warm-start from current color sampling.
3. **E-step** consistency (robust weight + TV/low-rank prior on $\mathcal M$).
4. Uncertainty output (§6) + uncertainty-driven completion.
5. Animate+relight transport (§8) in the runtime; differentiable for end-to-end.
6. Synthetic ground-truth harness (§10) — this is what turns claims into a paper.

Step 6 is built *first* alongside step 2, because the synthetic harness is how we know the math is
right before chasing real-photo polish.

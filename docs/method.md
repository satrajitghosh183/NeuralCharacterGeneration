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

> **Empirical support (2026-06-22→24, `tests/recon/test_inverse_render.cpp`, all green on H100):**
> - **C1 (identifiability):** scaled albedo error **0.069 (N=1) → 0.029 (N=5)** — recovery is good
>   under multi-illumination and *strictly improves with lighting diversity*, exactly as predicted.
> - **C2 (robustness):** with **35% corrupted** observations (clothing-swap/junk), the robust
>   consistency estimator gets **0.031 vs 0.096** for non-robust (≈3×), and the inferred
>   consistency is **0.004 on corrupted vs ≈1 on clean** — it *identifies* the junk.
> - **Relighting:** recovered albedo rendered under a **novel, never-observed** directional light
>   matches the ground-truth-albedo render to **6.9% relative error** — relightable, no light stage.
>
> Idealized so far (synthetic Lambertian normals); the real-geometry + error-vs-N sweep and the
> real-photo relighting render are the next experiments.

> **Benchmark on real SMPL-X geometry (V=10,475, `ncg_cli benchmark`, 2026-06-24):**
> | #photos N | albedo err | relight err (novel light) |
> |---|---|---|
> | 1 | 0.096 | 0.227 |
> | 3 | 0.058 | 0.132 |
> | 5 | 0.049 | 0.097 |
> | 8 | 0.035 | 0.073 |
> | 12 | 0.029 | 0.052 |
>
> | corruption | robust (C2) | naive |
> |---|---|---|
> | 0% | 0.023 | 0.015 |
> | 20% | 0.024 | 0.056 |
> | 35% | 0.028 | 0.082 |
> | 50% | 0.029 | 0.106 |
>
> Monotone improvement with lighting diversity (C1), and robustness stays ~flat while naive
> averaging degrades ~7× under corruption (C2). The small clean-data cost of robustness (0% row)
> is expected and motivates the auto-scaled kernel / "clean album ⇒ less down-weighting".

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

> **Empirical support (2026-06-24, `tests/recon/test_inverse_render.cpp`, green on H100):**
> `transport_normals` rotates a full-weight bone's normals exactly by $R$, and shading the posed
> normals under a world light equals shading canonical normals under the pulled-back light to
> **max error $2.4\times10^{-7}$** — animate and relight commute (the SH addition-theorem identity
> $\sum_m Y_{lm}(d)Y_{lm}(Rn)=\sum_m Y_{lm}(R^\top d)Y_{lm}(n)$ holds in code).

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

---

## 13. C4 — motion identity: the same principle on the pose manifold

The appearance half (C1/C2) says: *identity is the invariant that explains inconsistent
observations across nuisance diversity.* The motion half is the **same statement on the pose
manifold**, with **action** playing the role **lighting** played for appearance.

**Generative model.** A person $p$ has a latent **style** $z_p$ (gait phase, limb timing, posture,
hand idiosyncrasy). Each casual clip $c$ of them performing some **content** (action) $u_c(t)$
produces a pose sequence
$$\theta_{c}(t) = g\big(u_c(t),\, z_p\big) + \varepsilon ,$$
where $g$ composes content and style on $SO(3)^J$. Casual monocular video is noisy and cut-ridden,
so $\varepsilon$ is heavy-tailed with structured outliers — *exactly* the appearance setting.

**Recovery (robust factorization).** Estimate the shared $z_p$ and per-clip content $\{u_c\}$:
$$\min_{z_p,\{u_c\},\{w_{ct}\}} \sum_{c,t} w_{ct}\,\rho\!\big(\theta_c(t)-g(u_c(t),z_p)\big) + \Psi(w) + \mathcal R(z_p)+\mathcal R(u),$$
with the **same half-quadratic robust weights $w_{ct}$** as §3 — they down-weight cut frames and
bad NLF poses (the jumpy-casual-video problem turned into the contribution). Solved by alternation:
fix style → recover each clip's content (a per-clip fit); fix content → update the shared style
(pooled across clips); reweight $w$.

**Proposition (informal, parallel to §4).** $z_p$ is identifiable up to a gauge if $p$ is observed
across $\ge 2$ **distinct** actions (content not a common reparameterization), because the *shared*
style must explain *all* clips under *different* content — the cross-action constraint breaks the
style/content ambiguity that a single clip cannot. **Discovery:** the *action diversity* of a
casual album makes personal motion style identifiable, just as *lighting diversity* makes albedo
identifiable. Same theorem shape, different manifold.

**Transfer (the payoff).** Given a new action request $u^\star(t)$ (from gameplay) and the recovered
$z_p$: $\theta^\star(t)=g(u^\star(t),z_p)$ — that action **in their style**, in any scenario. The
style is an *intrinsic property* of the avatar, not a clip. (The real-time *generative* controller
that maps live game inputs → $u^\star$ is the systems extension; the factorization + identifiability
is the contribution.)

**C4 claims (to validate in code, mirroring C1/C2):** (a) style recovery improves with **action
diversity**; (b) **robust** factorization beats naive pooling on **noisy/cut** casual clips; (c)
recovered style **transfers** to a held-out action. Synthetic ground-truth harness: generate
motions with known $(z_p,\{u_c\})$, corrupt with noise + cut-frames, recover, measure — exactly the
appearance benchmark, on motion.

**Why this is the "fire."** It unifies *appearance* and *motion* personal identity under **one**
recovery principle, from **casual** data, with **identifiability** on both manifolds, rendered by a
**commuting animate+relight** real-time runtime. Prior motion-style work uses clean labeled mocap;
none recovers style from noisy casual video *and* unifies it with relightable appearance into one
personalized avatar. That intersection is new.

> **Empirical support (2026-06-25, `tests/recon/test_motion_style.cpp`, green on H100):** with the
> robust ALS solver, held-out-action transfer error is **0.004 at 6 actions vs 0.580 at 2 actions**
> — the full style is identifiable only with action diversity, exactly the C1 effect on the pose
> manifold. Under **30% cut/outlier frames**, robust = **0.004 vs naive 0.735** — robust recovery
> is essentially exact where naive pooling collapses. (Synthetic ground-truth factors; real-video
> + a generative real-time controller are the next steps.)

## 14. C5 — identity from incoherent observation (the hardest case, the win condition)

**Problem.** Given `N` casual images of one person that disagree in *many* nuisances at once —
lighting, pose, camera, **outfit, occlusion, even which person is framed** — recover a single
coherent, relightable identity. Standard multi-view/inverse rendering assumes a *consistent*
appearance; standard robust estimators assume a *dominant inlier mode*. Casual web data of one
person has **neither** — so naive averaging blurs and majority-vote robustness has no majority.

**Generative model (per-vertex, Lambertian under per-image SH light — the C1 model).** NLF gives,
for image `f`, a pose and camera, hence for each visible vertex `v` an observed color
`O_{f,v} ∈ ℝ³`, a posed world normal `n_{f,v}`, and a validity weight `π_{f,v} ∈ [0,1]`:

```
O_{f,v}  ≈  a_v ⊙ ( L_f · b(n_{f,v}) )                                   (1)
```

`a_v` = per-vertex albedo (the identity — **shared across all `f`**); `L_f ∈ ℝ^{3×9}` = image `f`'s
order-2 SH illumination (per channel, **unknown, per-image**); `b(n) ∈ ℝ⁹` = SH basis with the
Lambertian half-cosine convolution folded in; `⊙` = per-channel product.

**Estimator (block-coordinate, each block closed-form + CUDA-parallel; robust E-step).**
Minimize `Σ_{f,v} w_{f,v} ‖O_{f,v} − a_v ⊙ (L_f·b(n_{f,v}))‖²` with `w_{f,v}=π_{f,v}·ν_{f,v}`:
- **L-step** (per image): given `a`, solve the 9-per-channel SH normal equations (ridge `λ_L`).
- **A-step** (per vertex): given `{L_f}`, solve `a_v` in closed form (ridge `λ_a`).
- **E-step** (consistency, C2): `ν_{f,v} = exp(−‖r_{f,v}‖² / 2σ²)`, residual
  `r_{f,v}=O_{f,v} − a_v⊙(L_f·b(n_{f,v}))`, scale auto-set `σ = 1.4826·MAD(r)` (half-quadratic /
  Welsch IRLS). Iterate L→A→E. `a` is identifiable up to a per-channel gauge (global scale), fixed
  by matching `mean(a)` to the robust mean observed color.

**Why identity *is* recoverable from incoherent data (the crux).** The exposed identity surface —
**face, skin, hair** — is observed *consistently* across every image (same albedo, only the light
changes), so under (1) its residuals are small and `ν→1`: it forms the inlier set and pins `a`
there. Clothing changes per image, so its residuals are large and `ν→0`: it is *excluded* from `a`
rather than averaged into a blur. The recovered `a` is therefore the **identity albedo**, relightable
by `shade_sh(a, L, n)` under any chosen `L`, and kept correct when posed by normal transport (§8/C3).
Naming is deliberate: we recover **who they are**, not an outfit the data never agreed on — inventing
a non-existent consensus outfit would be hallucination, not reconstruction.

**The novelty — from *reject* to *decompose*.** Stack `O ∈ ℝ^{N×V×3}`. The shaded-identity term
`S_{f,v}=a_v⊙(L_f·b(n_{f,v}))` is *low-rank in appearance* (one shared `a`; a 9-D light per image);
the per-image nuisance `G_{f,v}=O−S` (garment/occlusion) is what each image adds. C2 (implemented)
treats `G` as outliers to **reject** (`ν→0`). The **proposed extension** treats `G` as a *structured
per-image residual* (sparse, or low-rank per garment) to **model and subtract** — recovering more
identity (e.g. skin under partial occlusion) and optionally a per-image garment field. This is a
**robust low-rank-identity + sparse-nuisance factorization under a per-image illumination operator**:
the precise, novel estimator for truly incoherent capture. It strictly generalizes both classical
multi-illumination photometric stereo (which assumes one light or one shot) and robust PCA (which has
no physical illumination/shading operator and no shared-albedo manifold constraint).

**Unification (the thesis closes).** The identical structure is C4 on the motion manifold (§13): a
shared low-rank **style** `W` (identity) explains many actions (each exercising a subspace) with
per-action **content** (nuisance), recovered by robust ALS. Appearance (C1/C2/C5) and motion (C4) are
the **same invariant-vs-nuisance factorization on two manifolds** — one principle, one avatar,
recovered from data too inconsistent for any averaging method. *That intersection is the contribution.*

> **Status.** Implemented + green: `solve_inverse_render` (L/A/E, auto-scaled Welsch,
> `tests/recon/test_inverse_render.cpp`) now drives the photoreal Gaussian avatar via
> `ncg_cli avatar --identity`. On a deliberately incoherent pile (≈150 mixed web frames of one
> subject: different films/outfits/lighting, montage cuts, occasional wrong person), the naive fit
> collapses to a spiky blur, while the identity solver — **rejecting ≈70% of observations as
> inconsistent (mean consistency ≈0.3)** — recovers a clean, complete, 360° relightable body. The
> garment-residual decomposition is the proposed next step.
>
> **Ablation that pins the claim (`--identity --refine`).** Re-introducing a *photometric* (per-pixel
> image) fit on even the most-consistent 50% of frames **re-blurs** the result (fit collapses, body
> spikes return) — direct evidence that on incoherent appearance *no* image-space optimization can
> converge, because the targets genuinely disagree. Only the **robust per-vertex albedo** (reject
> per-observation, never average) survives. This delimits where the contribution lives: recovery must
> happen in the **consistency-gated parameter domain (albedo on the manifold)**, not in image space.
> **Ceiling + true next step:** per-vertex albedo is inherently smooth (≈10⁴ verts) and SMPL-X face
> shape is near-template, so the recovered face is a clean *identity* but not yet a recognizable
> likeness. Sharpening it is not a photometric fit but a resolution+geometry upgrade: (i) lift the
> robust albedo from per-vertex to **per-texel (UV)** so the same C1/C2 solver recovers a high-res
> face texture, and (ii) **personalized face geometry** from the consistent face observations.
>
> **Update — per-texel UV albedo implemented (`avatar --identity --uv-texture T`).** The SMPL-X UV
> layout (`vt`/`ft`) is now exported + loaded; `recon::uv_rasterize` (tested) gives a per-texel
> barycentric correspondence, and the C1/C2 solver runs in **texel space** (T², e.g. 512² ≫ 10⁴
> verts). On the incoherent pile it recovers a high-res albedo texture whose face island shows real
> eye/nose/mouth structure — the resolution per-vertex couldn't hold. Shape is also robustly
> personalized (median of per-frame betas). Textured-glTF export of the UV map is implemented
> (`mesh::write_glb_textured`, seam-unwelded + embedded PNG + material), so the recovered face shows
> on the rigged character in any engine. **Remaining for full likeness:** fine **face geometry** —
> either a per-vertex displacement field or, more in keeping with the thesis, **per-texel photometric
> normals** solved from the same multi-illumination observations (photometric stereo on the UV map),
> which adds high-frequency facial detail as a normal map without leaving the robust parameter domain.

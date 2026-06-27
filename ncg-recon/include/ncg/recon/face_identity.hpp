#pragma once

#include <ncg/core/tensor.hpp>

namespace ncg::recon {

// ============================================================================================
// Joint neutral-identity / expression / illumination factorization — the paper's novel estimator
// (docs/method.md Theorem 2, restricted to the face). THIS IS THE CONTRIBUTION, not the front-end.
//
// Problem. A casual album shows one person across a COMPOUND nuisance: every photo has unknown SH
// lighting Lᵢ, unknown facial expression ψᵢ, and unknown head pose πᵢ. Existing tools fix all but
// one (relighting fixes pose+expr; face recon fixes light) and cannot recover the person's true
// NEUTRAL identity face — the face that no single photo shows — while simultaneously delivering a
// relightable skin albedo.
//
// Model.  Oᵢ(v) = Lᵢ · b(πᵢ ∘ n(β,ψᵢ; v))  ⊙  a(v)            (per face-vertex v, per photo i)
//   β    : SHARED neutral identity face shape   (the invariant X_face — what we recover)
//   ψᵢ   : per-photo expression                 (nuisance, factored out)
//   Lᵢ   : per-photo order-2 SH light           (nuisance; the C1 multi-illumination signal)
//   πᵢ   : per-photo head pose                  (nuisance)
//   a    : SHARED relightable albedo on the face manifold (recovered with β)
//   b    : Lambertian SH basis; n(β,ψ) the posed-shape normals on SMPL-X's FLAME head.
//
// Estimator (Theorem 2).  Robust Riemannian block-coordinate descent over M × Gᴺ:
//   • Shape step  : Gauss–Newton on β against multi-photo landmark + photometric residuals.
//   • Expr/pose   : per-photo Gauss–Newton on (ψᵢ, πᵢ).
//   • Light step  : per-photo closed-form SH solve (C1 L-step).
//   • Albedo step : per-vertex closed-form solve (C1 A-step).
//   • Consistency : per-observation half-quadratic field νᵢ(v) (C2) rejecting expression outliers,
//                   occlusion, wrong-person — its breakdown point grows with album diversity.
//   • Gauge fix   : canonical NEUTRAL expression (ψ̄=0), white-balanced albedo, frontal π — this is
//                   the stabilizer H of Theorem 1, fixed explicitly.
// The photometric residual self-aligns the face across photos (no external landmark detector); the
// 51 SMPL-X face landmarks seed it. Identity is the over-constrained invariant ⇒ recovers sharpest.
// ============================================================================================

struct FaceIdentityConfig {
  int iterations = 30;        // outer block-coordinate sweeps
  float id_ridge = 1e-2F;     // Tikhonov on β (identity prior)
  float expr_ridge = 1e-1F;   // stronger prior on per-photo expression (pulls toward neutral)
  bool robust = true;         // C2 per-photo consistency (rejects wrong-person / bad detections)
  float robust_k = 3.0F;      // Welsch scale = robust_k · median residual (auto)
};

/// Result of the identity/expression/pose factorization. `id_shape` is the recovered NEUTRAL
/// identity — the face no single photo shows — which is the invariant X_face of Theorem 2.
struct FaceIdentityResult {
  Tensor id_shape;     // [n_id]    neutral identity shape coefficients (the invariant)
  Tensor expr;         // [N,n_expr] per-photo expression (nuisance, factored out)
  Tensor scale;        // [N]       per-photo weak-perspective scale
  Tensor rot;          // [N,3,3]   per-photo head rotation
  Tensor trans;        // [N,2]     per-photo image translation
  Tensor weight;       // [N]       per-photo consistency (Theorem 2's ν, 1=trusted)
  double residual = 0; // final robust landmark reprojection RMS
};

/// Theorem 2 (shape core): recover the SHARED neutral identity face shape + per-photo expression &
/// weak-perspective pose from per-photo 2D landmarks, by robust block-coordinate least squares —
/// the identity/expression disentanglement that is the paper's novel kernel (no single photo shows
/// the neutral face; it is the invariant across the album's expression diversity, recoverable per
/// Theorem 1). `base` [L,3] mean landmark positions; `id_basis` [L,3,n_id] / `expr_basis`
/// [L,3,n_expr] are SMPL-X's FLAME shapedirs sampled at the L landmarks; `landmarks2d` [N,L,2].
/// Steps per sweep: per-photo (expr, pose) Gauss-Newton; shared-β linear solve stacking all photos;
/// robust per-photo reweight. Identity is over-constrained ⇒ recovers sharply.
FaceIdentityResult solve_face_identity(const Tensor& base, const Tensor& id_basis,
                                       const Tensor& expr_basis, const Tensor& landmarks2d,
                                       const FaceIdentityConfig& cfg);

}  // namespace ncg::recon

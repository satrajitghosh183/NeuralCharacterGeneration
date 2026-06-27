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
  int iterations = 40;          // outer block-coordinate sweeps
  int n_id = 100;               // identity face shape dims to solve (SMPL-X has ~300)
  int n_expr = 50;              // expression dims (SMPL-X has ~100)
  float shape_ridge = 1e-2F;    // Tikhonov on β (identity prior)
  float expr_ridge = 1e-1F;     // stronger prior on per-photo expression (toward neutral)
  bool robust = true;           // C2 consistency field
  float robust_scale = 0.0F;    // 0 => auto (1.4826·MAD)
};

struct FaceIdentityResult {
  Tensor id_shape;     // [n_id]      the recovered NEUTRAL identity face (the invariant X_face)
  Tensor expr;         // [N,n_expr]  per-photo expression (nuisance)
  Tensor lights;       // [N,3,9]     per-photo SH light (nuisance)
  Tensor albedo;       // [V_face,3]  relightable face albedo (recovered jointly)
  Tensor consistency;  // [N,V_face]  per-observation consistency ν (Theorem 2)
  double residual = 0; // final robust photometric residual
};

/// Solve the joint factorization above. `obs` [N,Vf,3] per-photo face-vertex colors, `normals`
/// [N,Vf,3] posed normals, `weights` [N,Vf] visibility; `id_basis` [Vf,3,n_id] + `expr_basis`
/// [Vf,3,n_expr] are SMPL-X's FLAME identity/expression shapedirs restricted to the face, and
/// `landmarks2d` [N,L,2] the per-photo seed landmarks. Recovers the neutral identity face + per-photo
/// nuisance. (Implementation is the next milestone — this header pins the math being built.)
FaceIdentityResult solve_face_identity(const Tensor& obs, const Tensor& normals,
                                       const Tensor& weights, const Tensor& id_basis,
                                       const Tensor& expr_basis, const Tensor& landmarks2d,
                                       const FaceIdentityConfig& cfg);

}  // namespace ncg::recon

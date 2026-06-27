#pragma once

#include <ncg/core/tensor.hpp>

#include <vector>

namespace ncg::geom {

// ============================================================================================
// PHASE B INTERFACE CONTRACT (frozen). The novel out-of-subspace geometry solver (docs/method.md
// §M4/§M5): recover the person's identity β PLUS a per-vertex displacement field Δv that leaves the
// SMPL-X shape subspace — while keeping the rig (Δv inherits each vertex's lbs_weights, joints stay
// on the prior). Δv is gated by an OBSERVABILITY field o(v): vertices seen from diverse angles by
// confident dense correspondences are solved; unobserved vertices are pinned (Δv=0) — the firewall
// that stops the solve hallucinating geometry on the unseen back of the head. o(v) is the boundary
// Phases E/F consume.
//
// Core solver works on explicit tensors (no PhotoBundle/asset dependency) so it is proven on
// SYNTHETIC ground truth (known off-subspace Δv → assert recovery on observed, ≈0 on unobserved)
// before touching real data. A thin wrapper (later) extracts these from PhotoBundle + SmplxModel +
// the FaceMesh→SMPL-X dense embedding.
// ============================================================================================

struct GeomConfig {
  int iterations = 25;        // outer block-coordinate sweeps
  float id_ridge = 1e-2F;     // Tikhonov on β
  float expr_ridge = 1e-1F;   // expression prior (toward neutral)
  float lap_weight = 50.0F;   // λ_lap — cotangent-Laplacian smoothness on Δv
  float mag_weight = 5.0F;    // λ_mag — magnitude prior (stay near SMPL-X)
  int cg_iters = 80;          // conjugate-gradient iterations for the Δv normal equations
  bool robust = true;         // C2 per-photo Welsch reweight
  float robust_k = 3.0F;
  float o_solve = 0.25F;      // HARD gate: Δv pinned to 0 where observability o(v) < o_solve
};

/// Recovered geometry. `delta_v` is the displacement that personalizes the face beyond SMPL-X's
/// PCA subspace (0 outside the observed region by construction). `obs` is the persisted
/// observability field consumed by completion (Phases E/F).
struct GeomResult {
  Tensor beta;       // [n_id] identity shape coefficients
  Tensor delta_v;    // [V,3]  per-vertex displacement (gated by observability)
  Tensor obs;        // [V]    observability o(v) ∈ [0,1]
  Tensor expr;       // [N,n_ex] per-photo expression (nuisance)
  Tensor weight;     // [N]    per-photo trust (C2)
  double residual = 0;
};

/// Theorem-2-with-Δv core. `base` [V,3] template verts; `id_basis` [V,3,n_id] / `expr_basis`
/// [V,3,n_ex] are SMPL-X shape/expression dirs; `faces` [F,3]; `lap` [V,V] cotangent Laplacian;
/// `assoc` [K] geometry-face index per dense point + `bary` [K,3] barycentric weights (the
/// FaceMesh→SMPL-X embedding); `landmarks2d` [N,K,2] per-photo dense 2D points; `conf` [N,K] γ;
/// `w_prior` [N] per-photo trust (from the Phase-A gate). Block-coordinate descent: per-photo pose
/// & expression (ridge), shared β (ridge), shared Δv (Laplacian-regularized normal equations via
/// matrix-free CG), robust reweight; then the observability field + hard gate.
GeomResult solve_geometry(const Tensor& base, const Tensor& id_basis, const Tensor& expr_basis,
                          const Tensor& faces, const Tensor& lap, const Tensor& assoc,
                          const Tensor& bary, const Tensor& landmarks2d, const Tensor& conf,
                          const Tensor& w_prior, const GeomConfig& cfg = {});

}  // namespace ncg::geom

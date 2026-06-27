#pragma once

#include <ncg/core/tensor.hpp>

namespace ncg::fit {

// ============================================================================================
// PHASE C INTERFACE CONTRACT (frozen). The render-only free-splat layer (Layer 2, docs/method.md
// §M9) is bound to the rig-bearing mesh (Layer 1) by a k-NN SOFT binding so it deforms WITH the
// body and does NOT swim relative to the surface under pose. Each free splat blends the FULL
// per-vertex transforms (which already carry Δv + lbs_weights) of its k nearest Layer-1 vertices:
//   B_s(τ) = Σ_{j∈N(s)} ω_{s,j} ( vertex_transform_j(τ) ),   ω_{s,j} ∝ exp(−‖μ_s−v_j‖²/h²)
// Feed B_s to deform_avatar to skin the splats. The ANTI-SWIM metric (first-class, benchmarked)
// extends test_gs_skinning from 1:1 (<1e-5) to k-NN free binding.
// ============================================================================================

/// k-NN soft binding of N free splats to the Layer-1 mesh vertices.
struct SplatBinding {
  Tensor idx;     // [N,k] int64 — the k nearest Layer-1 vertices per splat
  Tensor weight;  // [N,k] f32   — distance-Gaussian weights, rows sum to 1
};

/// Bind each splat center to its `k` nearest mesh vertices with weights ω ∝ exp(−d²/h²); the
/// bandwidth h is set per-splat from its k-NN distances (`h_scale`× the mean). `centers` [N,3],
/// `verts` [V,3]. (Brute-force k-NN — fine at test/avatar scale; a spatial index is the scale TODO.)
SplatBinding bind_splats_knn(const Tensor& centers, const Tensor& verts, int64_t k = 4,
                             float h_scale = 1.5F);

/// Blend the per-vertex rest→posed transforms `vertex_transforms` [V,4,4] into one transform per
/// splat [N,4,4] using the binding (the LBS-style linear blend). Pass the result to deform_avatar
/// (with an empty binding) to skin the free splats.
Tensor blend_vertex_transforms(const Tensor& vertex_transforms, const SplatBinding& binding);

/// Anti-swim metric (§M9): mean over splats of ‖ μ'_s − Φ(μ_s) ‖, where μ'_s = B_s·μ_s (splat
/// deformed by its blended transform) and Φ = Σ_j ω_{s,j} (transform_j · v_j) (the bound surface
/// points deformed individually, then blended). 0 when the splat sits exactly on a vertex (1:1);
/// small for k-NN; a spike means the binding is wrong. `centers_rest` [N,3], `verts_rest` [V,3],
/// `vertex_transforms` [V,4,4].
double swim_metric(const Tensor& centers_rest, const Tensor& verts_rest,
                   const Tensor& vertex_transforms, const SplatBinding& binding);

}  // namespace ncg::fit

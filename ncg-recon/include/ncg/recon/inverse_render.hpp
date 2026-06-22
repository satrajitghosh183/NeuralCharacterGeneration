#pragma once

#include <ncg/core/tensor.hpp>

namespace ncg::recon {

/// Order-2 spherical-harmonics basis with the Lambertian (half-cosine) convolution folded in, so
/// irradiance under SH lighting L ([.,9]) at a unit normal n is simply E(n) = L . sh_basis(n).
/// `normals`: [N,3] (need not be normalized; normalized internally). Returns [N,9].
Tensor sh_basis(const Tensor& normals);

/// Forward Lambertian SH shading. albedo [V,3], sh [3,9] per-channel coeffs, normals [V,3].
/// Returns rendered RGB [V,3] = albedo ⊙ (sh · sh_basis(normals)).
Tensor shade_sh(const Tensor& albedo, const Tensor& sh, const Tensor& normals);

struct InverseRenderConfig {
  int iterations = 40;       // L/A alternations
  float albedo_ridge = 1e-4F;
  float light_ridge = 1e-3F;  // stabilizes the 9x9 SH normal equations
};

/// Recovered canonical appearance + per-photo lighting + uncertainty (see docs/method.md §5–6).
struct InverseRenderResult {
  Tensor albedo;     // [V,3]  canonical albedo on the body manifold
  Tensor lights;     // [N,3,9] per-photo SH lighting (per channel)
  Tensor precision;  // [V,3]  per-vertex Gauss-Newton precision (∝ inverse posterior variance)
};

/// Multi-illumination inverse rendering (the paper's core, §3–§5 of docs/method.md): given the
/// SAME canonical albedo observed across N photos under DIFFERENT unknown SH lighting, jointly
/// recover albedo + per-photo lighting by closed-form block-coordinate descent (per-photo SH
/// L-step, per-vertex albedo A-step). `weights` are the visibility×consistency weights w·ν.
///   obs     : [N,V,3]  observed RGB per photo per vertex (0 where unseen)
///   normals : [N,V,3]  posed world normals per photo per vertex
///   weights : [N,V]    in [0,1] (visibility × consistency); 0 drops the observation
/// This is the estimator that replaces the naive weighted-average fuse_vertex_colors; albedo is
/// identifiable up to one global per-channel scale (gauge), so compare/normalize accordingly.
InverseRenderResult solve_inverse_render(const Tensor& obs, const Tensor& normals,
                                         const Tensor& weights, const InverseRenderConfig& cfg);

}  // namespace ncg::recon

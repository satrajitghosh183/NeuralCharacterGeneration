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

/// SH coefficients [3,9] for a directional light from `direction` ([3]) of `color` ([3]) plus an
/// ambient term. Lets us relight a recovered albedo under any chosen lighting (the payoff of the
/// inverse-rendering decomposition): shade_sh(albedo, sh_directional_light(d,c,a), normals).
Tensor sh_directional_light(const Tensor& direction, const Tensor& color, float ambient = 0.2F);

/// Transport canonical surface normals into a posed frame under linear-blend skinning:
///   n_out[v] = normalize( sum_j skin_weights[v,j] * bone_rotations[j] @ normals_can[v] ).
/// This is the C3 contribution (docs/method.md §8): the shading frame must rotate with the bones,
/// or relighting becomes wrong once the avatar is posed — the reason naive splat avatars can't be
/// animated AND relit. With correctly transported normals, relight and animate commute.
///   normals_can [V,3], skin_weights [V,J], bone_rotations [J,3,3].
Tensor transport_normals(const Tensor& normals_can, const Tensor& skin_weights,
                         const Tensor& bone_rotations);

struct InverseRenderConfig {
  int iterations = 40;       // L/A alternations
  float albedo_ridge = 1e-4F;
  float light_ridge = 1e-3F;  // stabilizes the 9x9 SH normal equations
  bool robust = true;         // infer a per-observation consistency weight (C2) to reject
                              // inconsistent observations (clothing swaps, occlusion, junk)
  float robust_scale = 0.0F;  // Welsch kernel scale; 0 => auto (1.4826 * MAD of residuals)
};

/// Recovered canonical appearance + per-photo lighting + uncertainty (see docs/method.md §5–6).
struct InverseRenderResult {
  Tensor albedo;       // [V,3]  canonical albedo on the body manifold
  Tensor lights;       // [N,3,9] per-photo SH lighting (per channel)
  Tensor precision;    // [V,3]  per-vertex Gauss-Newton precision (∝ inverse posterior variance)
  Tensor consistency;  // [N,V]  inferred per-observation consistency weight (1=trusted, 0=rejected)
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

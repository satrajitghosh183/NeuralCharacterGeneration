#pragma once

#include <ncg/core/tensor.hpp>
#include <ncg/recon/gaussian_model.hpp>

namespace ncg::recon {

/// Per-vertex isotropic Gaussian scale from local vertex spacing: the mean distance to each
/// vertex's `k` nearest neighbours, times `mult`. Dense regions (fingers, face) get small
/// splats and sparse regions (torso) larger ones — fixes the "comet"/over-spray artifact a
/// single global scale produces on a non-uniform mesh. Returns [V] (clamped > 0).
/// O(V^2) via cdist — fine for SMPL-X (~10k verts) on the GPU as a one-time init.
Tensor per_vertex_scale(const Tensor& vertices, double mult = 0.75, int k = 3);

/// Initializes one Gaussian per body vertex (the Phase-1 slice initializer — no optimization,
/// no fusion). Rotation identity; opacity 1.
///   vertices         : [V,3] world-space SMPL-X vertices
///   scale            : isotropic world-space std-dev (used when per_vertex_scale_t is undefined)
///   colors           : [V,3] linear RGB in [0,1], or undefined for a default gray
///   per_vertex_scale_t : optional [V] per-vertex std-dev (e.g. from per_vertex_scale()); when
///                        defined it overrides `scale`
GaussianCloud gaussians_on_body(const Tensor& vertices, float scale, const Tensor& colors = {},
                                const Tensor& per_vertex_scale_t = {});

}  // namespace ncg::recon

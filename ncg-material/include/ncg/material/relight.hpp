#pragma once

#include <ncg/core/tensor.hpp>

#include <array>

namespace ncg::material {

/// Analytical Lambertian + ambient relighting (a real, non-learned shader):
///   out = albedo * (ambient + max(0, n . l) * light_color)
/// All images are CHW; `normal_chw` are world-space unit normals in [-1,1]. This is the
/// shading model for the relightable path; obtaining (albedo, normal) from a single image
/// (intrinsic decomposition / IDArb) is the gated learned step — see decompose() in intrinsic.hpp.
Tensor relight(const Tensor& albedo_chw, const Tensor& normal_chw,
               std::array<float, 3> light_dir, std::array<float, 3> light_color = {1.0F, 1.0F, 1.0F},
               float ambient = 0.1F);

}  // namespace ncg::material

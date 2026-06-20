#include <ncg/material/relight.hpp>

#include <ncg/core/error.hpp>

#include <cmath>

namespace ncg::material {

Tensor relight(const Tensor& albedo_chw, const Tensor& normal_chw, std::array<float, 3> light_dir,
               std::array<float, 3> light_color, float ambient) {
  NCG_CHECK(albedo_chw.dim() == 3 && albedo_chw.size(0) == 3, "relight: albedo must be [3,H,W]");
  NCG_CHECK(normal_chw.sizes() == albedo_chw.sizes(), "relight: normal must match albedo shape");

  const auto opts = albedo_chw.options();
  const auto n = normal_chw.to(at::kFloat);
  const auto n_unit = n / n.norm(2, /*dim=*/0, /*keepdim=*/true).clamp_min(1e-8);  // [3,H,W]

  const float ln = std::sqrt(light_dir[0] * light_dir[0] + light_dir[1] * light_dir[1] +
                             light_dir[2] * light_dir[2]);
  const float inv = (ln > 1e-8F) ? 1.0F / ln : 0.0F;
  const auto l = torch::tensor({light_dir[0] * inv, light_dir[1] * inv, light_dir[2] * inv}, opts)
                     .view({3, 1, 1});
  const auto lc = torch::tensor({light_color[0], light_color[1], light_color[2]}, opts)
                      .view({3, 1, 1});

  const auto ndotl = (n_unit * l).sum(/*dim=*/0, /*keepdim=*/true).clamp_min(0.0);  // [1,H,W]
  return albedo_chw.to(at::kFloat) * (ambient + ndotl * lc);
}

}  // namespace ncg::material

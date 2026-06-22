#include <ncg/recon/appearance.hpp>

#include <ncg/core/error.hpp>

#include <torch/torch.h>

namespace ncg::recon {

Tensor sample_vertex_colors(const Tensor& image_chw, const Tensor& verts2d) {
  NCG_CHECK(image_chw.dim() == 3 && image_chw.size(0) == 3,
            "sample_vertex_colors: image must be [3,H,W], got {}-D", image_chw.dim());
  NCG_CHECK(verts2d.dim() == 2 && verts2d.size(1) == 2,
            "sample_vertex_colors: verts2d must be [V,2]");

  const auto opts = image_chw.options();
  const auto H = static_cast<double>(image_chw.size(1));
  const auto W = static_cast<double>(image_chw.size(2));
  const auto img = image_chw.unsqueeze(0);  // [1,3,H,W]

  const auto p = verts2d.to(opts);
  const auto x = p.select(1, 0);  // column
  const auto y = p.select(1, 1);  // row
  // Normalize pixel coords -> [-1,1] for grid_sample (align_corners=true => exact at centers).
  const auto gx = x / (W - 1.0) * 2.0 - 1.0;
  const auto gy = y / (H - 1.0) * 2.0 - 1.0;
  const auto grid = torch::stack({gx, gy}, 1).view({1, -1, 1, 2});  // [1,V,1,2]

  namespace F = torch::nn::functional;
  const auto sampled = F::grid_sample(
      img, grid,
      F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kBorder).align_corners(
          true));  // [1,3,V,1]
  return sampled.squeeze(3).squeeze(0).transpose(0, 1).contiguous();  // [V,3]
}

}  // namespace ncg::recon

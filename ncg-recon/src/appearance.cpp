#include <ncg/recon/appearance.hpp>

#include <ncg/core/error.hpp>

#include <torch/torch.h>

#include <limits>

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
  // Border padding is emulated by CLAMPING the grid to [-1,1] (with align_corners=true, ±1 hits
  // the edge pixel exactly — identical to kBorder) because MPS does not implement kBorder.
  const auto gx = (x / (W - 1.0) * 2.0 - 1.0).clamp(-1.0, 1.0);
  const auto gy = (y / (H - 1.0) * 2.0 - 1.0).clamp(-1.0, 1.0);
  const auto grid = torch::stack({gx, gy}, 1).view({1, -1, 1, 2});  // [1,V,1,2]

  namespace F = torch::nn::functional;
  const auto sampled = F::grid_sample(
      img, grid,
      F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(
          true));  // [1,3,V,1]
  return sampled.squeeze(3).squeeze(0).transpose(0, 1).contiguous();  // [V,3]
}

Tensor vertex_visibility(const Tensor& verts2d, const Tensor& depth, int64_t height,
                         int64_t width, double depth_tol) {
  NCG_CHECK(verts2d.dim() == 2 && verts2d.size(1) == 2, "vertex_visibility: verts2d must be [V,2]");
  NCG_CHECK(depth.dim() == 1 && depth.size(0) == verts2d.size(0),
            "vertex_visibility: depth must be [V] matching verts2d");

  const auto opts_f = depth.options().dtype(at::kFloat);
  const auto d = depth.to(at::kFloat);
  // Pixel index per vertex (nearest pixel, clamped into the image).
  const auto col = verts2d.select(1, 0).round().to(at::kLong).clamp(0, width - 1);
  const auto row = verts2d.select(1, 1).round().to(at::kLong).clamp(0, height - 1);
  const auto idx = row * width + col;  // [V] in [0, H*W)

  // Point z-buffer: frontmost (min) depth per pixel.
  auto zbuf = torch::full({height * width}, std::numeric_limits<float>::infinity(), opts_f);
  zbuf.scatter_reduce_(0, idx, d, "amin", /*include_self=*/true);
  const auto front = zbuf.gather(0, idx);  // [V] frontmost depth at each vertex's pixel
  return (d <= front + static_cast<float>(depth_tol)).to(at::kFloat);
}

FusedAppearance fuse_vertex_colors(const std::vector<Tensor>& colors,
                                   const std::vector<Tensor>& weights) {
  NCG_CHECK(!colors.empty(), "fuse_vertex_colors: no views");
  NCG_CHECK(colors.size() == weights.size(), "fuse_vertex_colors: colors/weights count mismatch");

  const auto opts = colors.front().options();
  const int64_t V = colors.front().size(0);
  auto acc_color = torch::zeros({V, 3}, opts);
  auto acc_w = torch::zeros({V}, opts);
  for (size_t i = 0; i < colors.size(); ++i) {
    NCG_CHECK(colors[i].size(0) == V && colors[i].size(1) == 3,
              "fuse_vertex_colors: view {} colors must be [V,3]", i);
    NCG_CHECK(weights[i].size(0) == V, "fuse_vertex_colors: view {} weights must be [V]", i);
    const auto w = weights[i].to(opts);
    acc_color = acc_color + colors[i].to(opts) * w.unsqueeze(1);
    acc_w = acc_w + w;
  }

  FusedAppearance out;
  const auto denom = acc_w.clamp_min(1e-8).unsqueeze(1);
  out.colors = acc_color / denom;
  // Vertices seen in no view get a neutral gray so they read as "unknown", not black.
  const auto unseen = (acc_w <= 0).unsqueeze(1);
  out.colors = torch::where(unseen, torch::full_like(out.colors, 0.5F), out.colors);
  out.coverage = acc_w;
  return out;
}

}  // namespace ncg::recon

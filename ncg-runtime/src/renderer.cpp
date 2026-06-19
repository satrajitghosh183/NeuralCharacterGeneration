#include <ncg/runtime/renderer.hpp>

#include <ncg/core/error.hpp>
#include <ncg/runtime/splat_raster.hpp>

#include <algorithm>

namespace ncg::runtime {

RenderOutput render_gaussians(const recon::GaussianCloud& g, const Camera& cam,
                              std::array<float, 3> background) {
  g.validate();
  NCG_CHECK(g.device().is_cuda(), "render_gaussians: cloud must be on CUDA");
  NCG_CHECK(cam.width > 0 && cam.height > 0, "render_gaussians: invalid image size");

  Tensor uv;
  Tensor depth;
  cam.project(g.positions, uv, depth);  // uv [N,2], depth [N]

  const auto u = uv.select(1, 0).contiguous();
  const auto v = uv.select(1, 1).contiguous();

  // Screen-space std-dev from world scale and depth; floor at half a pixel.
  const auto mean_scale = g.scales.mean(/*dim=*/1);                                  // [N]
  const auto sigma_px = (cam.fx * mean_scale / depth.clamp_min(1e-3)).clamp_min(0.5);  // [N]
  const auto inv_s2 = (1.0 / (sigma_px * sigma_px)).to(at::kFloat).contiguous();

  // Cull Gaussians at/behind the image plane by zeroing their opacity.
  const auto in_front = (depth > 0.01).to(at::kFloat);
  const auto op = (g.opacities.squeeze(1) * in_front).contiguous();

  // Composite near -> far.
  const auto order = depth.argsort(/*dim=*/0, /*descending=*/false);
  auto sel = [&](const Tensor& x) { return x.index_select(0, order).contiguous(); };

  auto [image, alpha] = splat_render_cuda(sel(u), sel(v), sel(inv_s2), sel(op),
                                          sel(g.colors), cam.height, cam.width, background);
  return {image, alpha};
}

RenderOutput render_soft(const recon::GaussianCloud& g, const Camera& cam,
                         std::array<float, 3> background, int64_t chunk) {
  g.validate();
  NCG_CHECK(cam.width > 0 && cam.height > 0, "render_soft: invalid image size");
  const int H = cam.height;
  const int W = cam.width;
  const auto opts = g.positions.options();

  Tensor uv;
  Tensor depth;
  cam.project(g.positions, uv, depth);  // differentiable

  const auto mean_scale = g.scales.mean(/*dim=*/1);                                  // [N]
  const auto sigma_px = (cam.fx * mean_scale / depth.clamp_min(1e-3)).clamp_min(0.5);  // [N]
  const auto inv_s2 = 1.0 / (sigma_px * sigma_px);                                     // [N]
  const auto in_front = (depth > 0.01).to(at::kFloat).detach();                        // mask
  const auto op = g.opacities.squeeze(1) * in_front;                                   // [N]

  const auto xs = (torch::arange(W, opts) + 0.5).view({1, 1, W});  // [1,1,W]
  const auto ys = (torch::arange(H, opts) + 0.5).view({1, H, 1});  // [1,H,1]

  auto wsum = torch::zeros({H, W}, opts);
  auto csum = torch::zeros({3, H, W}, opts);

  const int64_t n = g.positions.size(0);
  for (int64_t s = 0; s < n; s += chunk) {
    const int64_t e = std::min(s + chunk, n);
    using torch::indexing::Slice;
    const auto cu = uv.index({Slice(s, e), 0}).view({-1, 1, 1});       // [C,1,1]
    const auto cv = uv.index({Slice(s, e), 1}).view({-1, 1, 1});       // [C,1,1]
    const auto dx = xs - cu;                                          // [C,1,W]
    const auto dy = ys - cv;                                          // [C,H,1]
    const auto d2 = dx * dx + dy * dy;                                // [C,H,W]
    const auto w = op.index({Slice(s, e)}).view({-1, 1, 1}) *
                   torch::exp(-0.5 * d2 * inv_s2.index({Slice(s, e)}).view({-1, 1, 1}));  // [C,H,W]
    wsum = wsum + w.sum(0);
    csum = csum + torch::einsum("chw,ck->khw", {w, g.colors.index({Slice(s, e)})});  // [3,H,W]
  }

  const auto coverage = (1.0 - torch::exp(-wsum)).unsqueeze(0);  // [1,H,W] soft alpha
  const auto fg = csum / (wsum.unsqueeze(0) + 1e-8);
  const auto bg = torch::tensor({background[0], background[1], background[2]}, opts).view({3, 1, 1});
  RenderOutput out;
  out.image = fg * coverage + bg * (1.0 - coverage);
  out.alpha = coverage;
  return out;
}

}  // namespace ncg::runtime

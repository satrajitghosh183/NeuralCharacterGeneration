#include <ncg/runtime/renderer.hpp>

#include <ncg/core/error.hpp>
#include <ncg/runtime/splat_raster.hpp>

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

}  // namespace ncg::runtime

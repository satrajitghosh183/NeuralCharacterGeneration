#include <catch2/catch_test_macros.hpp>

#include <ncg/core/device.hpp>
#include <ncg/nerf/nerf.hpp>
#include <ncg/record/metrics.hpp>
#include <ncg/runtime/camera.hpp>

#include <torch/torch.h>

#include <vector>

TEST_CASE("camera_rays are unit-length and count H*W", "[nerf]") {
  const auto center = torch::zeros({3});
  const auto cam = ncg::runtime::Camera::orbit(center, 2.0F, 0.0F, 0.0F, 50.0F, 16, 12, at::kCPU);
  auto [origins, dirs] = ncg::nerf::camera_rays(cam);
  REQUIRE(dirs.size(0) == 16 * 12);
  REQUIRE(origins.size(0) == 16 * 12);
  const auto norms = dirs.norm(2, 1);
  REQUIRE(torch::allclose(norms, torch::ones_like(norms), 1e-4, 1e-4));
}

TEST_CASE("render_volume returns a finite image of the right shape", "[nerf]") {
  ncg::nerf::NerfConfig cfg;
  cfg.samples = 16;
  ncg::nerf::TinyNerf nerf(cfg);
  const auto cam = ncg::runtime::Camera::orbit(torch::zeros({3}), 2.0F, 0.0F, 0.0F, 50.0F, 16, 16,
                                               at::kCPU);
  const auto out = ncg::nerf::render_volume(nerf, cam);
  REQUIRE(out.image.sizes() == (std::vector<int64_t>{3, 16, 16}));
  REQUIRE(out.alpha.sizes() == (std::vector<int64_t>{1, 16, 16}));
  REQUIRE(std::isfinite(out.image.sum().item<double>()));
}

TEST_CASE("composite_over (premultiplied alpha): opaque hides back, transparent shows back",
          "[nerf]") {
  // composite_over uses the premultiplied-alpha "over" operator, matching what render_volume
  // and the splat renderer emit (image = sum of weighted color, already premultiplied). Under
  // that convention a transparent pixel's premultiplied color is 0, an opaque pixel's is its
  // color, and a half-covered pixel's is 0.5*color.
  ncg::runtime::RenderOutput back;
  back.image = torch::full({3, 4, 4}, 0.9F);
  back.alpha = torch::ones({1, 4, 4});

  // Opaque front (alpha=1, premult image = color) fully hides the back.
  ncg::runtime::RenderOutput front;
  front.image = torch::full({3, 4, 4}, 0.2F);
  front.alpha = torch::ones({1, 4, 4});
  auto opaque = ncg::nerf::composite_over(front, back);
  REQUIRE(torch::allclose(opaque.image, front.image, 1e-6, 1e-6));

  // Transparent front (alpha=0 => premult color 0) shows the back unchanged.
  front.image = torch::zeros({3, 4, 4});
  front.alpha = torch::zeros({1, 4, 4});
  auto clear = ncg::nerf::composite_over(front, back);
  REQUIRE(torch::allclose(clear.image, back.image, 1e-6, 1e-6));

  // Half-covered front (alpha=0.5, premult color = 0.5*color) blends: 0.5*c + 0.5*back.
  const auto c = 0.4F;
  front.image = torch::full({3, 4, 4}, 0.5F * c);
  front.alpha = torch::full({1, 4, 4}, 0.5F);
  auto blend = ncg::nerf::composite_over(front, back);
  const auto expected = torch::full({3, 4, 4}, 0.5F * c + 0.5F * 0.9F);
  REQUIRE(torch::allclose(blend.image, expected, 1e-6, 1e-6));
}

TEST_CASE("NeRF fit overfits a single view below the gray baseline", "[nerf]") {
  if (!ncg::cuda_available()) {
    SKIP("CUDA not available");
  }
  const auto dev = at::Device(at::kCUDA, 0);
  const auto opts = at::TensorOptions().dtype(at::kFloat).device(dev);
  const int W = 24;
  const int H = 24;

  // A horizontal gradient target (non-constant, so the gray baseline is beatable).
  auto ramp = torch::linspace(0, 1, W, opts).view({1, 1, W}).expand({3, H, W}).contiguous();
  const auto cam = ncg::runtime::Camera::orbit(torch::zeros({3}, opts), 2.0F, 0.0F, 0.0F, 50.0F, W,
                                               H, dev);

  ncg::nerf::NerfConfig nc;
  nc.samples = 24;
  ncg::nerf::NerfFitConfig fc;
  fc.iterations = 120;
  fc.log_every = 1000;
  auto nerf = ncg::nerf::fit_nerf_to_views({ramp}, {cam}, nc, fc, nullptr);

  const auto rendered = ncg::nerf::render_volume(*nerf, cam).image;
  const double fit_psnr = ncg::record::psnr(rendered, ramp);
  const double gray_psnr = ncg::record::psnr(torch::full_like(ramp, 0.5), ramp);
  INFO("fit_psnr=" << fit_psnr << " gray_psnr=" << gray_psnr);
  REQUIRE(fit_psnr > gray_psnr);
}

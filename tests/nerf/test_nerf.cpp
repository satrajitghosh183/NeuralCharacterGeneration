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

TEST_CASE("composite_over: opaque front hides back; transparent front shows back", "[nerf]") {
  ncg::runtime::RenderOutput front;
  ncg::runtime::RenderOutput back;
  front.image = torch::full({3, 4, 4}, 0.2F);
  back.image = torch::full({3, 4, 4}, 0.9F);
  back.alpha = torch::ones({1, 4, 4});

  front.alpha = torch::ones({1, 4, 4});  // opaque
  auto opaque = ncg::nerf::composite_over(front, back);
  REQUIRE(torch::allclose(opaque.image, front.image, 1e-6, 1e-6));

  front.alpha = torch::zeros({1, 4, 4});  // transparent
  auto clear = ncg::nerf::composite_over(front, back);
  REQUIRE(torch::allclose(clear.image, back.image, 1e-6, 1e-6));
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

#include <catch2/catch_test_macros.hpp>

#include <ncg/core/device.hpp>
#include <ncg/fit/fit_image.hpp>
#include <ncg/record/metrics.hpp>
#include <ncg/recon/gaussian_model.hpp>
#include <ncg/runtime/camera.hpp>
#include <ncg/runtime/renderer.hpp>

#include <torch/torch.h>

// Optimization sanity: fitting Gaussians to a target image must improve PSNR. The target is
// itself a soft-rendered cloud, so a solution exists. Validates render_soft gradients +
// the Adam loop end to end.

TEST_CASE("fitting a Gaussian cloud to a target image reduces error", "[cuda][fit]") {
  if (!ncg::cuda_available()) {
    SKIP("CUDA not available");
  }
  const auto dev = at::Device(at::kCUDA, 0);
  const auto opts = at::TensorOptions().dtype(at::kFloat).device(dev);
  const int W = 48;
  const int H = 48;

  const auto cam = ncg::runtime::Camera::orbit(torch::zeros({3}, opts), 2.5F, 0.0F, 0.0F, 50.0F,
                                               W, H, dev);

  // Build a target by soft-rendering a small fixed cloud.
  torch::manual_seed(0);
  ncg::recon::GaussianCloud tgt;
  tgt.positions = torch::randn({40, 3}, opts) * 0.25;
  tgt.scales = torch::full({40, 3}, 0.05F, opts);
  tgt.rotations = torch::zeros({40, 4}, opts);
  tgt.rotations.select(1, 0).fill_(1.0);
  tgt.opacities = torch::full({40, 1}, 0.8F, opts);
  tgt.colors = torch::rand({40, 3}, opts);
  const auto target = ncg::runtime::render_soft(tgt, cam).image.detach();

  ncg::fit::FitConfig cfg;
  cfg.iterations = 60;
  cfg.num_gaussians = 200;
  cfg.log_every = 1000;  // quiet

  // Baseline PSNR from a fresh random init (1 render).
  const auto fitted = ncg::fit::fit_gaussians_to_image(target, cam, cfg, nullptr);
  const auto rendered = ncg::runtime::render_soft(fitted, cam).image;
  const double final_psnr = ncg::record::psnr(rendered, target);

  // A trivial gray image is the no-op baseline; fitting must beat it comfortably.
  const double gray_psnr = ncg::record::psnr(torch::full_like(target, 0.5), target);
  INFO("final_psnr=" << final_psnr << " gray_psnr=" << gray_psnr);
  REQUIRE(final_psnr > gray_psnr);
}

#include <catch2/catch_test_macros.hpp>

#include <ncg/core/device.hpp>
#include <ncg/fit/fit_image.hpp>
#include <ncg/record/metrics.hpp>
#include <ncg/recon/gaussian_model.hpp>
#include <ncg/runtime/camera.hpp>
#include <ncg/runtime/renderer.hpp>

#include <torch/torch.h>

#include <vector>

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

TEST_CASE("multi-view fitting beats the gray baseline on held views", "[cuda][fit]") {
  if (!ncg::cuda_available()) {
    SKIP("CUDA not available");
  }
  const auto dev = at::Device(at::kCUDA, 0);
  const auto opts = at::TensorOptions().dtype(at::kFloat).device(dev);
  const int W = 40;
  const int H = 40;

  torch::manual_seed(1);
  ncg::recon::GaussianCloud tgt;
  tgt.positions = torch::randn({40, 3}, opts) * 0.25;
  tgt.scales = torch::full({40, 3}, 0.05F, opts);
  tgt.rotations = torch::zeros({40, 4}, opts);
  tgt.rotations.select(1, 0).fill_(1.0);
  tgt.opacities = torch::full({40, 1}, 0.8F, opts);
  tgt.colors = torch::rand({40, 3}, opts);

  std::vector<ncg::runtime::Camera> cams;
  std::vector<at::Tensor> targets;
  for (float az : {-30.0F, 30.0F}) {
    auto cam = ncg::runtime::Camera::orbit(torch::zeros({3}, opts), 2.5F, az, 0.0F, 50.0F, W, H, dev);
    targets.push_back(ncg::runtime::render_soft(tgt, cam).image.detach());
    cams.push_back(cam);
  }

  ncg::fit::FitConfig cfg;
  cfg.iterations = 60;
  cfg.num_gaussians = 200;
  cfg.log_every = 1000;
  const auto fitted = ncg::fit::fit_gaussians_to_views(targets, cams, cfg, nullptr);

  for (size_t v = 0; v < cams.size(); ++v) {
    const auto rendered = ncg::runtime::render_soft(fitted, cams[v]).image;
    const double fit_psnr = ncg::record::psnr(rendered, targets[v]);
    const double gray_psnr = ncg::record::psnr(torch::full_like(targets[v], 0.5), targets[v]);
    REQUIRE(fit_psnr > gray_psnr);
  }
}

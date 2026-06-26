#include <catch2/catch_test_macros.hpp>

#include <ncg/core/device.hpp>
#include <ncg/fit/fit_adaptive.hpp>
#include <ncg/record/metrics.hpp>
#include <ncg/recon/gaussian_model.hpp>
#include <ncg/runtime/camera.hpp>
#include <ncg/runtime/renderer.hpp>

#include <torch/torch.h>

#include <vector>

// Adaptive 3DGS fit: from a SPARSE seed, adaptive density control + anisotropic splatting +
// D-SSIM must reconstruct a richer target cloud across several views, growing the Gaussian count
// and beating both the gray baseline and a fixed-count isotropic fit at equal seed size.

namespace {
ncg::recon::GaussianCloud random_cloud(int64_t n, const at::TensorOptions& opts, uint64_t seed) {
  torch::manual_seed(seed);
  ncg::recon::GaussianCloud g;
  g.positions = torch::randn({n, 3}, opts) * 0.25;
  g.scales = torch::full({n, 3}, 0.05F, opts);
  g.rotations = torch::zeros({n, 4}, opts);
  g.rotations.select(1, 0).fill_(1.0);
  g.opacities = torch::full({n, 1}, 0.7F, opts);
  g.colors = torch::rand({n, 3}, opts);
  g.validate();
  return g;
}
}  // namespace

TEST_CASE("ssim is 1 for identical images and lower for corrupted", "[fit][adaptive]") {
  const auto opts = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  const auto a = torch::rand({3, 32, 32}, opts);
  REQUIRE(ncg::fit::ssim(a, a).item<double>() > 0.999);
  const auto b = (a + 0.4 * torch::randn_like(a)).clamp(0.0, 1.0);
  REQUIRE(ncg::fit::ssim(a, b).item<double>() < 0.95);
}

TEST_CASE("adaptive fit densifies and reconstructs a target from a sparse seed", "[cuda][fit]") {
  const auto dev = ncg::cuda_available() ? at::Device(at::kCUDA, 0) : at::Device(at::kCPU);
  const auto opts = at::TensorOptions().dtype(at::kFloat).device(dev);
  const int W = 48;
  const int H = 48;

  // Rich target cloud -> render several views as supervision.
  const auto tgt = random_cloud(120, opts, /*seed=*/7);
  std::vector<ncg::runtime::Camera> cams;
  std::vector<at::Tensor> targets;
  for (float az : {-25.0F, 0.0F, 25.0F, 60.0F}) {
    auto cam = ncg::runtime::Camera::orbit(torch::zeros({3}, opts), 2.5F, az, 0.0F, 50.0F, W, H, dev);
    targets.push_back(ncg::runtime::render_soft_aniso(tgt, cam).image.detach());
    cams.push_back(cam);
  }

  // Sparse seed: far fewer Gaussians than the target.
  const auto seed = random_cloud(30, opts, /*seed=*/99);

  ncg::fit::AdaptiveFitConfig cfg;
  cfg.iterations = 500;
  cfg.densify_from = 50;
  cfg.densify_until = 400;
  cfg.densify_every = 50;
  cfg.opacity_reset_every = 0;  // keep the small test deterministic
  cfg.use_mask = false;         // synthetic targets have no separate background
  cfg.log_every = 1000;         // quiet
  const auto fitted = ncg::fit::fit_adaptive(targets, cams, seed, cfg, nullptr);

  // Densification must have grown the cloud past the seed.
  INFO("seed N=" << seed.size() << " fitted N=" << fitted.size());
  REQUIRE(fitted.size() > seed.size());

  // Reconstruction beats the gray baseline on every supervised view.
  double mean_psnr = 0.0;
  for (size_t v = 0; v < cams.size(); ++v) {
    const auto rendered = ncg::runtime::render_soft_aniso(fitted, cams[v]).image;
    const double fit_psnr = ncg::record::psnr(rendered, targets[v]);
    const double gray_psnr = ncg::record::psnr(torch::full_like(targets[v], 0.5), targets[v]);
    INFO("view " << v << " fit_psnr=" << fit_psnr << " gray_psnr=" << gray_psnr);
    REQUIRE(fit_psnr > gray_psnr);
    mean_psnr += fit_psnr;
  }
  REQUIRE(mean_psnr / static_cast<double>(cams.size()) > 18.0);
}

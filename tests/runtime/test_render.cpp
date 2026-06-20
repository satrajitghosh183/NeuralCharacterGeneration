#include <catch2/catch_test_macros.hpp>

#include <ncg/core/device.hpp>
#include <ncg/recon/gaussian_model.hpp>
#include <ncg/runtime/camera.hpp>
#include <ncg/runtime/renderer.hpp>

#include <torch/torch.h>

// Smoke test for the forward splatter: a single opaque white Gaussian at the origin should
// render with non-trivial coverage and a bright center.

TEST_CASE("forward splat renders a centered blob", "[cuda][runtime]") {
  if (!ncg::cuda_available()) {
    SKIP("CUDA not available");
  }
  const auto dev = at::Device(at::kCUDA, 0);
  const auto opts = at::TensorOptions().dtype(at::kFloat).device(dev);

  ncg::recon::GaussianCloud g;
  g.positions = torch::zeros({1, 3}, opts);
  g.scales = torch::full({1, 3}, 0.15F, opts);
  g.rotations = torch::tensor({{1.0F, 0.0F, 0.0F, 0.0F}}, opts);
  g.opacities = torch::ones({1, 1}, opts);
  g.colors = torch::ones({1, 3}, opts);
  g.validate();

  const auto center = torch::zeros({3}, opts);
  const auto cam = ncg::runtime::Camera::orbit(center, /*radius=*/1.5F, 0.0F, 0.0F,
                                               /*fov_y_deg=*/50.0F, 64, 64, dev);

  const auto out = ncg::runtime::render_gaussians(g, cam, {0.0F, 0.0F, 0.0F});
  REQUIRE(out.image.sizes() == (std::vector<int64_t>{3, 64, 64}));
  REQUIRE(out.alpha.max().item<double>() > 0.1);
  REQUIRE(std::isfinite(out.image.sum().item<double>()));

  // Center pixel should be brighter than a corner.
  const double center_lum = out.image.index({0, 32, 32}).item<double>();
  const double corner_lum = out.image.index({0, 0, 0}).item<double>();
  REQUIRE(center_lum > corner_lum);
}

TEST_CASE("orbit_trajectory yields N distinct cameras", "[runtime]") {
  const auto center = torch::zeros({3});
  const auto cams = ncg::runtime::orbit_trajectory(center, 2.0F, 10.0F, /*frames=*/8, 50.0F, 32, 32,
                                                   at::kCPU);
  REQUIRE(cams.size() == 8);
  // Opposite frames must look from different positions.
  REQUIRE_FALSE(torch::allclose(cams[0].t, cams[4].t));
}

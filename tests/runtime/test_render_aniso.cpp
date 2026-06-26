#include <catch2/catch_test_macros.hpp>

#include <ncg/recon/gaussian_model.hpp>
#include <ncg/runtime/camera.hpp>
#include <ncg/runtime/renderer.hpp>

#include <torch/torch.h>

// Anisotropic EWA splatter (render_soft_aniso): runs on CPU. Verifies (1) a centered blob renders
// finite with a bright center, (2) anisotropic scale produces an oriented (elongated) footprint
// that an isotropic disk cannot, and (3) gradients flow to the rotation quaternion — the property
// render_soft lacks and the reason oriented splats can be learned.

namespace {
ncg::recon::GaussianCloud one_gaussian(const at::TensorOptions& opts, at::Tensor scales,
                                       at::Tensor quat) {
  ncg::recon::GaussianCloud g;
  g.positions = torch::zeros({1, 3}, opts);
  g.scales = scales;
  g.rotations = quat;
  g.opacities = torch::ones({1, 1}, opts);
  g.colors = torch::ones({1, 3}, opts);
  g.validate();
  return g;
}
const auto kOpts = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
ncg::runtime::Camera front_cam() {
  // Camera looking down -z onto the origin from +z, so the x/y Gaussian axes map to image x/y.
  return ncg::runtime::Camera::orbit(torch::zeros({3}), /*radius=*/2.0F, 0.0F, 0.0F,
                                     /*fov_y_deg=*/50.0F, 64, 64, at::kCPU);
}
}  // namespace

TEST_CASE("aniso splat: centered blob renders finite with bright center", "[runtime][aniso]") {
  const auto g = one_gaussian(kOpts, torch::full({1, 3}, 0.12F, kOpts),
                              torch::tensor({{1.0F, 0.0F, 0.0F, 0.0F}}, kOpts));
  const auto out = ncg::runtime::render_soft_aniso(g, front_cam(), {0.0F, 0.0F, 0.0F});
  REQUIRE(out.image.sizes() == (std::vector<int64_t>{3, 64, 64}));
  REQUIRE(std::isfinite(out.image.sum().item<double>()));
  REQUIRE(out.alpha.max().item<double>() > 0.3);
  REQUIRE(out.image.index({0, 32, 32}).item<double>() >
          out.image.index({0, 2, 2}).item<double>());
}

TEST_CASE("aniso splat: anisotropic scale yields an oriented footprint", "[runtime][aniso]") {
  // A Gaussian wide in world-x, thin in world-y -> alpha should spread farther horizontally than
  // vertically. An isotropic renderer (mean scale) collapses this distinction.
  const auto g = one_gaussian(kOpts, torch::tensor({{0.30F, 0.05F, 0.05F}}, kOpts),
                              torch::tensor({{1.0F, 0.0F, 0.0F, 0.0F}}, kOpts));
  const auto out = ncg::runtime::render_soft_aniso(g, front_cam(), {0.0F, 0.0F, 0.0F});
  const auto alpha = out.alpha.squeeze(0);  // [H,W]
  const double thr = 0.05;
  // Horizontal extent at the center row vs vertical extent at the center column.
  const auto row = (alpha.index({32}) > thr).to(at::kFloat).sum().item<double>();
  const auto col = (alpha.index({torch::indexing::Slice(), 32}) > thr).to(at::kFloat).sum().item<double>();
  REQUIRE(row > col * 1.8);  // clearly elongated horizontally
}

TEST_CASE("aniso splat: gradients flow to the rotation quaternion", "[runtime][aniso]") {
  auto quat = torch::tensor({{1.0F, 0.05F, 0.02F, 0.0F}}, kOpts).set_requires_grad(true);
  auto g = one_gaussian(kOpts, torch::tensor({{0.30F, 0.06F, 0.06F}}, kOpts), quat);
  const auto out = ncg::runtime::render_soft_aniso(g, front_cam(), {0.0F, 0.0F, 0.0F});
  // Asymmetric target so rotating the elongated splat changes the loss.
  const auto target = torch::zeros_like(out.image);
  target.index_put_({torch::indexing::Slice(), torch::indexing::Slice(20, 44), 50}, 1.0F);
  const auto loss = torch::mse_loss(out.image, target);
  loss.backward();
  REQUIRE(quat.grad().defined());
  REQUIRE(quat.grad().abs().sum().item<double>() > 1e-6);
}

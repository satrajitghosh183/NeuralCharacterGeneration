#include <catch2/catch_test_macros.hpp>

#include <ncg/fit/fit_avatar.hpp>
#include <ncg/recon/gaussian_model.hpp>

#include <torch/torch.h>

// deform_avatar skinning math, validated deterministically without a SMPL-X asset: identity
// transforms leave the cloud unchanged, and a shared rigid transform maps every Gaussian exactly
// by R·p + t while keeping orientations unit-norm. This is the pose-conditioning core of the
// animatable avatar (fit_avatar's full loop is exercised by the real-data run).

namespace {
ncg::recon::GaussianCloud make_cloud(int64_t n, const at::TensorOptions& opts) {
  torch::manual_seed(3);
  ncg::recon::GaussianCloud g;
  g.positions = torch::randn({n, 3}, opts);
  g.scales = torch::full({n, 3}, 0.05F, opts);
  g.rotations = torch::zeros({n, 4}, opts);
  g.rotations.select(1, 0).fill_(1.0);
  g.opacities = torch::full({n, 1}, 0.9F, opts);
  g.colors = torch::rand({n, 3}, opts);
  g.validate();
  return g;
}
}  // namespace

TEST_CASE("deform_avatar: identity transform is a no-op", "[fit][avatar]") {
  const auto opts = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  const auto c = make_cloud(5, opts);
  const auto eye = torch::eye(4, opts).unsqueeze(0).expand({5, 4, 4}).contiguous();
  const auto posed = ncg::fit::deform_avatar(c, eye);
  REQUIRE(torch::allclose(posed.positions, c.positions, 1e-5, 1e-5));
  REQUIRE(torch::allclose(posed.rotations, c.rotations, 1e-5, 1e-5));
  REQUIRE(torch::allclose(posed.colors, c.colors));
}

TEST_CASE("deform_avatar: shared rigid transform maps every Gaussian by R*p + t", "[fit][avatar]") {
  const auto opts = at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
  const int64_t n = 8;
  const auto c = make_cloud(n, opts);

  // 90° rotation about z, translation (1,2,3).
  const auto R = torch::tensor({{0.0F, -1.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 1.0F}}, opts);
  const auto t = torch::tensor({1.0F, 2.0F, 3.0F}, opts);
  auto vt = torch::zeros({n, 4, 4}, opts);
  using torch::indexing::Slice;
  vt.index_put_({Slice(), Slice(0, 3), Slice(0, 3)}, R.unsqueeze(0));
  vt.index_put_({Slice(), Slice(0, 3), 3}, t.unsqueeze(0));
  vt.index_put_({Slice(), 3, 3}, 1.0F);

  const auto posed = ncg::fit::deform_avatar(c, vt);
  const auto expected = torch::matmul(c.positions, R.t()) + t;  // (R p) + t per row
  REQUIRE(torch::allclose(posed.positions, expected, 1e-4, 1e-4));

  // Orientations stay unit quaternions and actually changed (a 90° twist was applied).
  const auto qnorm = posed.rotations.norm(2, 1);
  REQUIRE(torch::allclose(qnorm, torch::ones_like(qnorm), 1e-4, 1e-4));
  REQUIRE_FALSE(torch::allclose(posed.rotations, c.rotations, 1e-3, 1e-3));
}

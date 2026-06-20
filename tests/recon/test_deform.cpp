#include <catch2/catch_test_macros.hpp>

#include <ncg/recon/animate.hpp>
#include <ncg/recon/gaussian_model.hpp>

#include <torch/torch.h>

namespace {

ncg::recon::GaussianCloud make_cloud(int64_t n) {
  const auto opts = torch::TensorOptions().dtype(torch::kFloat);
  ncg::recon::GaussianCloud g;
  g.positions = torch::randn({n, 3}, opts);
  g.scales = torch::full({n, 3}, 0.02F, opts);
  g.rotations = torch::zeros({n, 4}, opts);
  g.rotations.select(1, 0).fill_(1.0);
  g.opacities = torch::ones({n, 1}, opts);
  g.colors = torch::rand({n, 3}, opts);
  g.validate();
  return g;
}

}  // namespace

TEST_CASE("deform by identity leaves positions unchanged", "[recon]") {
  auto g = make_cloud(50);
  const auto T = torch::eye(4).view({1, 4, 4}).expand({50, 4, 4}).contiguous();
  const auto out = ncg::recon::deform_gaussians(g, T);
  REQUIRE(torch::allclose(out.positions, g.positions, 1e-6, 1e-6));
}

TEST_CASE("deform by translation shifts positions", "[recon]") {
  auto g = make_cloud(50);
  auto T = torch::eye(4).view({1, 4, 4}).expand({50, 4, 4}).clone();
  T.index_put_({torch::indexing::Slice(), 0, 3}, 1.0);
  T.index_put_({torch::indexing::Slice(), 1, 3}, -2.0);
  T.index_put_({torch::indexing::Slice(), 2, 3}, 3.0);
  const auto out = ncg::recon::deform_gaussians(g, T);
  const auto delta = out.positions - g.positions;
  const auto expected = torch::tensor({1.0F, -2.0F, 3.0F}).view({1, 3}).expand_as(delta);
  REQUIRE(torch::allclose(delta, expected, 1e-5, 1e-5));
}

TEST_CASE("deform by 90deg z-rotation rotates positions", "[recon]") {
  // Rotation by +90deg about z maps (x,y,z) -> (-y, x, z).
  auto g = make_cloud(10);
  auto R = torch::zeros({3, 3});
  R[0][1] = -1.0;
  R[1][0] = 1.0;
  R[2][2] = 1.0;
  auto T = torch::eye(4);
  T.index_put_({torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)}, R);
  T = T.view({1, 4, 4}).expand({10, 4, 4}).contiguous();

  const auto out = ncg::recon::deform_gaussians(g, T);
  const auto px = g.positions.select(1, 0);
  const auto py = g.positions.select(1, 1);
  REQUIRE(torch::allclose(out.positions.select(1, 0), -py, 1e-5, 1e-5));
  REQUIRE(torch::allclose(out.positions.select(1, 1), px, 1e-5, 1e-5));
}

#include <catch2/catch_test_macros.hpp>

#include <ncg/recon/init_from_body.hpp>

#include <torch/torch.h>

#include <vector>

// Three points on a line at x = 0, 1, 3. With k=1 the nearest-neighbour distances are
// {1, 1, 2}, so per_vertex_scale (mult=1) returns exactly those — denser points get smaller
// scale than the isolated one.
TEST_CASE("per_vertex_scale tracks local spacing", "[recon][init]") {
  const auto verts = torch::tensor({{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, {3.0F, 0.0F, 0.0F}});
  const auto s = ncg::recon::per_vertex_scale(verts, /*mult=*/1.0, /*k=*/1);
  REQUIRE(torch::allclose(s, torch::tensor({1.0F, 1.0F, 2.0F}), 1e-5, 1e-5));
  REQUIRE(s[2].item<float>() > s[0].item<float>());  // isolated vertex gets a bigger splat
}

TEST_CASE("gaussians_on_body accepts a per-vertex scale", "[recon][init]") {
  const auto verts = torch::tensor({{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, {3.0F, 0.0F, 0.0F}});
  const auto pvs = ncg::recon::per_vertex_scale(verts, 1.0, 1);
  const auto cloud = ncg::recon::gaussians_on_body(verts, /*scale=*/0.01F, /*colors=*/{}, pvs);
  REQUIRE(cloud.scales.sizes() == (std::vector<int64_t>{3, 3}));
  // Per-vertex scale overrode the scalar: row 2 (isolated) is larger than row 0.
  REQUIRE(cloud.scales[2][0].item<float>() > cloud.scales[0][0].item<float>());
  REQUIRE(cloud.scales[0][0].item<float>() == cloud.scales[0][1].item<float>());  // isotropic
}

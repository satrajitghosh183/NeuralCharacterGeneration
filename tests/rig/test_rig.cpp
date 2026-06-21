#include <catch2/catch_test_macros.hpp>

#include <ncg/rig/rig.hpp>

#include <torch/torch.h>

#include <filesystem>

TEST_CASE("make_rigged validates and packages a rig", "[rig]") {
  const int64_t V = 12;
  const int64_t J = 3;
  const int64_t F = 4;
  auto verts = torch::randn({V, 3});
  auto faces = torch::randint(0, V, {F, 3}, torch::kLong);
  auto joints = torch::randn({J, 3});
  auto parents = torch::zeros({J}, torch::kLong);
  auto weights = torch::softmax(torch::randn({V, J}), 1);

  const auto m = ncg::rig::make_rigged(verts, faces, joints, parents, weights);
  REQUIRE(m.vertices.size(0) == V);
  REQUIRE(m.skin_weights.sizes() == (std::vector<int64_t>{V, J}));

  REQUIRE_THROWS(ncg::rig::make_rigged(verts, faces, joints, parents, torch::randn({V, J + 1})));
}

TEST_CASE("export_rigged writes OBJ + rig JSON", "[rig]") {
  const int64_t V = 6;
  const int64_t J = 2;
  auto m = ncg::rig::make_rigged(torch::randn({V, 3}), torch::randint(0, V, {2, 3}, torch::kLong),
                                 torch::randn({J, 3}), torch::zeros({J}, torch::kLong),
                                 torch::softmax(torch::randn({V, J}), 1));

  const auto base = (std::filesystem::temp_directory_path() / "ncg_rig").string();
  ncg::rig::export_rigged(m, base);
  REQUIRE(std::filesystem::file_size(base + ".obj") > 0);
  REQUIRE(std::filesystem::file_size(base + ".rig.json") > 0);
  std::filesystem::remove(base + ".obj");
  std::filesystem::remove(base + ".rig.json");
}

TEST_CASE("autorig (UniRig) is gated", "[rig]") {
  REQUIRE_THROWS(ncg::rig::autorig(torch::randn({4, 3}), torch::zeros({2, 3}, torch::kLong)));
}

TEST_CASE("transfer_skinning copies the nearest source weights", "[rig]") {
  // Two source verts with one-hot weights for 2 joints.
  const auto sv = torch::tensor({{0.0F, 0.0F, 0.0F}, {10.0F, 0.0F, 0.0F}});
  const auto sw = torch::tensor({{1.0F, 0.0F}, {0.0F, 1.0F}});
  // Targets near each source.
  const auto tv = torch::tensor({{0.1F, 0.0F, 0.0F}, {9.0F, 0.0F, 0.0F}, {0.0F, 0.2F, 0.0F}});

  const auto w = ncg::rig::transfer_skinning(tv, sv, sw);
  REQUIRE(w.sizes() == (std::vector<int64_t>{3, 2}));
  REQUIRE(w[0][0].item<float>() == 1.0F);  // near source 0
  REQUIRE(w[1][1].item<float>() == 1.0F);  // near source 1
  REQUIRE(w[2][0].item<float>() == 1.0F);  // near source 0
}

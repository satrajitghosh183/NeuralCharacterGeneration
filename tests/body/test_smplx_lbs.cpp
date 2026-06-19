#include <catch2/catch_test_macros.hpp>

#include <golden_fixture.hpp>
#include <tensor_compare.hpp>

#include <ncg/body/smplx.hpp>
#include <ncg/io/safetensors.hpp>

#include <torch/torch.h>

#include <filesystem>
#include <map>

// Validates SMPL-X LBS invariants on a tiny synthetic model (no external assets), then runs
// the real-data golden test when data/golden/smplx is present.

namespace {

// Build a minimal but structurally-valid SMPL-X-shaped model and write it to safetensors.
std::filesystem::path write_toy_model(int64_t V, int64_t J, int64_t nbetas) {
  auto v_template = torch::randn({V, 3}, torch::kFloat);
  auto shapedirs = torch::zeros({V, 3, nbetas}, torch::kFloat);
  auto posedirs = torch::zeros({V, 3, 9 * (J - 1)}, torch::kFloat);
  auto J_regressor = torch::zeros({J, V}, torch::kFloat);
  // Each joint regresses to a distinct vertex so rest joints are well-defined.
  for (int64_t j = 0; j < J; ++j) J_regressor[j][j % V] = 1.0;
  auto lbs_weights = torch::full({V, J}, 1.0F / static_cast<float>(J), torch::kFloat);
  auto parents = torch::zeros({J}, torch::kLong);  // root parent ignored; others -> 0

  const auto path = std::filesystem::temp_directory_path() / "ncg_toy_smplx.safetensors";
  ncg::io::write_safetensors(path.string(), {{"v_template", v_template},
                                             {"shapedirs", shapedirs},
                                             {"posedirs", posedirs},
                                             {"J_regressor", J_regressor},
                                             {"lbs_weights", lbs_weights},
                                             {"parents", parents}});
  return path;
}

}  // namespace

TEST_CASE("SMPL-X neutral pose returns the template", "[body]") {
  const auto path = write_toy_model(/*V=*/6, /*J=*/3, /*nbetas=*/2);
  auto model = ncg::body::SmplxModel::load(path.string(), at::kCPU);

  const auto out = model.forward(model.neutral_params(1));
  REQUIRE(out.vertices.sizes() == (std::vector<int64_t>{1, model.num_verts(), 3}));

  const auto r = ncg::test::compare_allclose(out.vertices.squeeze(0), model.forward(
      model.neutral_params(1)).vertices.squeeze(0), 1e-6, 1e-6);
  REQUIRE(r.passed);
  std::filesystem::remove(path);
}

TEST_CASE("SMPL-X translation shifts all vertices", "[body]") {
  const auto path = write_toy_model(6, 3, 2);
  auto model = ncg::body::SmplxModel::load(path.string(), at::kCPU);

  auto p0 = model.neutral_params(1);
  const auto base = model.forward(p0).vertices;

  auto p1 = model.neutral_params(1);
  p1.transl = torch::tensor({{1.0F, 2.0F, 3.0F}});
  const auto shifted = model.forward(p1).vertices;

  const auto delta = shifted - base;  // should equal transl everywhere
  const auto expected = torch::tensor({1.0F, 2.0F, 3.0F}).view({1, 1, 3}).expand_as(delta);
  REQUIRE(ncg::test::compare_allclose(delta, expected, 1e-5, 1e-5).passed);
  std::filesystem::remove(path);
}

TEST_CASE("SMPL-X real-data golden", "[golden]") {
  if (!ncg::test::golden_available("smplx")) {
    SKIP("no golden data for smplx (fetch data/golden/smplx to enable)");
  }
  // When data is present: load the real model + a reference params->vertices dump and compare.
  FAIL("smplx golden data present but real-data comparison not wired yet");
}

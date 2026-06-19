#include <catch2/catch_test_macros.hpp>

#include <ncg/io/npy.hpp>

#include <torch/torch.h>

#include <filesystem>

namespace {
std::filesystem::path temp_file(const char* name) {
  return std::filesystem::temp_directory_path() / name;
}
}  // namespace

TEST_CASE("npy round-trips float32 and int64", "[io]") {
  const auto f = torch::randn({2, 3, 4}, torch::kFloat);
  const auto pf = temp_file("ncg_npy_f.npy");
  ncg::io::save_npy(pf.string(), f);
  const auto rf = ncg::io::load_npy(pf.string());
  REQUIRE(rf.sizes() == f.sizes());
  REQUIRE(rf.scalar_type() == at::kFloat);
  REQUIRE(torch::allclose(rf, f));
  std::filesystem::remove(pf);

  const auto i = torch::randint(0, 100, {7}, torch::kLong);
  const auto pi = temp_file("ncg_npy_i.npy");
  ncg::io::save_npy(pi.string(), i);
  const auto ri = ncg::io::load_npy(pi.string());
  REQUIRE(ri.scalar_type() == at::kLong);
  REQUIRE(torch::equal(ri, i));
  std::filesystem::remove(pi);
}

TEST_CASE("npy handles 1-D and scalar shapes", "[io]") {
  const auto v = torch::tensor({1.5F, 2.5F, 3.5F});
  const auto p = temp_file("ncg_npy_v.npy");
  ncg::io::save_npy(p.string(), v);
  REQUIRE(torch::allclose(ncg::io::load_npy(p.string()), v));
  std::filesystem::remove(p);
}

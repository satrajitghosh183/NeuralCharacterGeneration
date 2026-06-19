#include <catch2/catch_test_macros.hpp>

#include <ncg/io/safetensors.hpp>

#include <torch/torch.h>

#include <filesystem>
#include <map>

namespace {
std::filesystem::path temp_file(const char* name) {
  return std::filesystem::temp_directory_path() / name;
}
}  // namespace

TEST_CASE("safetensors round-trips tensors and metadata", "[io]") {
  const auto a = torch::arange(12, torch::kFloat).reshape({3, 4});
  const auto b = torch::tensor({1, 2, 3}, torch::kLong);

  const auto path = temp_file("ncg_st_roundtrip.safetensors");
  ncg::io::write_safetensors(path.string(), {{"a", a}, {"b", b}}, {{"note", "hi"}});

  auto st = ncg::io::SafeTensors::open(path.string());
  REQUIRE(st.has("a"));
  REQUIRE(st.has("b"));
  REQUIRE_FALSE(st.has("missing"));
  REQUIRE(st.metadata().at("note") == "hi");

  const auto av = st.view("a");
  REQUIRE(av.sizes() == a.sizes());
  REQUIRE(av.scalar_type() == at::kFloat);
  REQUIRE(torch::equal(av, a));

  const auto bv = st.view("b");
  REQUIRE(bv.scalar_type() == at::kLong);
  REQUIRE(torch::equal(bv, b));

  std::filesystem::remove(path);
}

TEST_CASE("safetensors view of missing tensor throws", "[io]") {
  const auto path = temp_file("ncg_st_missing.safetensors");
  ncg::io::write_safetensors(path.string(), {{"x", torch::zeros({2})}});
  auto st = ncg::io::SafeTensors::open(path.string());
  REQUIRE_THROWS(st.view("nope"));
  std::filesystem::remove(path);
}

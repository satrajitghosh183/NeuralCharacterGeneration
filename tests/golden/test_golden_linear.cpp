#include <catch2/catch_test_macros.hpp>

#include <golden_fixture.hpp>
#include <tensor_compare.hpp>

#include <ncg/io/safetensors.hpp>
#include <ncg/io/weight_loader.hpp>

#include <torch/torch.h>

#include <filesystem>
#include <map>

// Harness self-test: exercises the ENTIRE weight-port path (write safetensors -> mmap read ->
// WeightMap::load_into a torch::nn::Module -> forward -> compare) on a trivial linear model,
// with NO committed binary fixtures. If a real model's golden test fails, run this first to
// isolate "harness bug" from "port bug".

namespace {

std::filesystem::path temp_file(const char* name) {
  return std::filesystem::temp_directory_path() / name;
}

}  // namespace

TEST_CASE("golden harness round-trips a linear layer", "[golden]") {
  torch::manual_seed(7);
  const int64_t in = 5;
  const int64_t out = 3;

  // Known reference parameters.
  const auto W = torch::randn({out, in}, torch::kFloat);
  const auto b = torch::randn({out}, torch::kFloat);

  const auto path = temp_file("ncg_golden_linear.safetensors");
  ncg::io::write_safetensors(path.string(), {{"weight", W}, {"bias", b}},
                             {{"__ncg__", "linear smoke test"}});

  // Port into a fresh module via the loader.
  torch::nn::Linear linear(torch::nn::LinearOptions(in, out).bias(true));
  ncg::io::WeightMap wm(ncg::io::SafeTensors::open(path.string()));
  wm.load_into(*linear, /*remap=*/{}, /*strict=*/true);

  // Forward vs the analytic reference.
  const auto x = torch::randn({4, in}, torch::kFloat);
  const auto actual = linear->forward(x);
  const auto expected = torch::addmm(b, x, W.t());

  const auto r = ncg::test::compare_allclose(actual, expected, 1e-5, 1e-6);
  INFO(r.summary());
  REQUIRE(r.passed);

  std::filesystem::remove(path);
}

TEST_CASE("WeightMap strict mode catches an unconsumed tensor", "[golden]") {
  const int64_t in = 2;
  const int64_t out = 2;
  const auto path = temp_file("ncg_golden_extra.safetensors");
  ncg::io::write_safetensors(path.string(), {{"weight", torch::randn({out, in})},
                                             {"bias", torch::randn({out})},
                                             {"unexpected", torch::randn({3})}});

  torch::nn::Linear linear(torch::nn::LinearOptions(in, out));
  ncg::io::WeightMap wm(ncg::io::SafeTensors::open(path.string()));
  REQUIRE_THROWS(wm.load_into(*linear, {}, /*strict=*/true));

  std::filesystem::remove(path);
}

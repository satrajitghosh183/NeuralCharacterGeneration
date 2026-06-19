#include <catch2/catch_test_macros.hpp>

#include <ncg/io/image.hpp>

#include <torch/torch.h>

#include <filesystem>

namespace {
std::filesystem::path temp_file(const char* name) {
  return std::filesystem::temp_directory_path() / name;
}
}  // namespace

TEST_CASE("PNG save/load round-trips within 8-bit quantization", "[io]") {
  // Deterministic gradient image, CHW in [0,1].
  const int64_t h = 16;
  const int64_t w = 24;
  auto img = torch::zeros({3, h, w}, torch::kFloat);
  img[0] = torch::linspace(0, 1, w).unsqueeze(0).expand({h, w});
  img[1] = torch::linspace(0, 1, h).unsqueeze(1).expand({h, w});
  img[2] = 0.5;

  const auto path = temp_file("ncg_img.png");
  ncg::io::save_png(path.string(), img);
  const auto loaded = ncg::io::load_image(path.string(), 3);

  REQUIRE(loaded.sizes() == img.sizes());
  // 8-bit quantization tolerance.
  REQUIRE(torch::allclose(loaded, img, /*rtol=*/0.0, /*atol=*/1.0 / 255.0 + 1e-4));
  std::filesystem::remove(path);
}

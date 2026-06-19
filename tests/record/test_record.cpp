#include <catch2/catch_test_macros.hpp>

#include <ncg/record/metrics.hpp>
#include <ncg/record/recorder.hpp>

#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <string>

TEST_CASE("metrics behave sensibly", "[record]") {
  torch::manual_seed(1);
  const auto a = torch::rand({3, 32, 32});
  const auto noisy = (a + 0.05 * torch::randn_like(a)).clamp(0, 1);
  const auto very_noisy = (a + 0.3 * torch::randn_like(a)).clamp(0, 1);

  REQUIRE(ncg::record::mae(a, a) == 0.0);
  REQUIRE(ncg::record::psnr(a, a) > 90.0);                 // identical -> capped high
  REQUIRE(ncg::record::ssim(a, a) > 0.99);                 // identical -> ~1
  REQUIRE(ncg::record::psnr(a, noisy) > ncg::record::psnr(a, very_noisy));
  REQUIRE(ncg::record::ssim(a, noisy) > ncg::record::ssim(a, very_noisy));
}

TEST_CASE("recorder writes a structured run directory", "[record]") {
  const auto root = (std::filesystem::temp_directory_path() / "ncg_rec_test").string();
  auto rec = ncg::record::Recorder::create(root, "unit", /*unique_suffix=*/true);

  REQUIRE(std::filesystem::exists(rec.dir()));
  REQUIRE(std::filesystem::exists(rec.dir() / "images"));
  REQUIRE(std::filesystem::exists(rec.dir() / "checkpoints"));

  rec.set_config("{\"k\":1}");
  rec.log_scalar("stageA", "loss", 0.5);
  rec.log_text("stageA", "note", "hello");
  { auto t = rec.time("stageA", "work"); }
  rec.log_image("render", "rgb", torch::rand({3, 8, 8}));

  REQUIRE(std::filesystem::exists(rec.dir() / "config" / "config.json"));
  REQUIRE(std::filesystem::exists(rec.dir() / "metrics.jsonl"));

  std::ifstream is(rec.dir() / "metrics.jsonl");
  int lines = 0;
  std::string line;
  while (std::getline(is, line)) {
    if (!line.empty()) ++lines;
  }
  REQUIRE(lines >= 4);  // loss + note + work_ms + image entry

  bool found_png = false;
  for (const auto& e : std::filesystem::directory_iterator(rec.dir() / "images")) {
    if (e.path().extension() == ".png") found_png = true;
  }
  REQUIRE(found_png);

  std::filesystem::remove_all(rec.dir());
}

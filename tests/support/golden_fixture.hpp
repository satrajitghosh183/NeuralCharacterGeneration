#pragma once

#include <ncg/core/error.hpp>
#include <ncg/io/npy.hpp>

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace ncg::test {

/// One intermediate to check during per-layer parity (matches a dump from dump_golden.py).
struct GoldenStage {
  std::string name;
  std::string ref_file;  // .npy under the model dir
  double rtol;
  double atol;
};

/// Parsed data/golden/<model>/manifest.json.
struct GoldenManifest {
  std::string model;
  std::string weights;  // .safetensors / .ncgw filename (under the model dir)
  std::string input;    // input .npy filename
  double rtol = 1e-3;
  double atol = 1e-4;
  std::vector<GoldenStage> stages;
};

/// Absolute path to data/golden/<model> in the source tree (NCG_SOURCE_DIR is compile-defined).
inline std::filesystem::path golden_dir(const std::string& model) {
  return std::filesystem::path(NCG_SOURCE_DIR) / "data" / "golden" / model;
}

/// True if the golden data for `model` is present (so tests can SKIP rather than fail when
/// the large reference dumps / weights have not been fetched).
inline bool golden_available(const std::string& model) {
  return std::filesystem::exists(golden_dir(model) / "manifest.json");
}

inline GoldenManifest load_manifest(const std::string& model) {
  const auto path = golden_dir(model) / "manifest.json";
  std::ifstream is(path);
  NCG_CHECK(is.good(), "golden: cannot open manifest '{}'", path.string());
  nlohmann::json j;
  is >> j;

  GoldenManifest m;
  m.model = j.value("model", model);
  m.weights = j.at("weights").get<std::string>();
  m.input = j.at("input").get<std::string>();
  if (j.contains("tolerance")) {
    m.rtol = j["tolerance"].value("rtol", m.rtol);
    m.atol = j["tolerance"].value("atol", m.atol);
  }
  for (const auto& s : j.at("stages")) {
    GoldenStage st;
    st.name = s.at("name").get<std::string>();
    st.ref_file = s.at("ref").get<std::string>();
    st.rtol = s.value("rtol", m.rtol);
    st.atol = s.value("atol", m.atol);
    m.stages.push_back(st);
  }
  return m;
}

inline at::Tensor load_golden(const std::string& model, const std::string& file) {
  return ncg::io::load_npy((golden_dir(model) / file).string());
}

}  // namespace ncg::test

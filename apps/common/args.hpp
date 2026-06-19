#pragma once

#include <ncg/core/error.hpp>

#include <string>
#include <unordered_map>

namespace ncg::app {

/// Minimal `--key value` / `--flag` command-line parser (no external dependency).
class Args {
public:
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string tok = argv[i];
      if (tok.rfind("--", 0) != 0) continue;
      const std::string key = tok.substr(2);
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        kv_[key] = argv[++i];
      } else {
        kv_[key] = "true";  // bare flag
      }
    }
  }

  bool has(const std::string& k) const { return kv_.find(k) != kv_.end(); }

  std::string get(const std::string& k, const std::string& def = "") const {
    auto it = kv_.find(k);
    return it == kv_.end() ? def : it->second;
  }
  std::string require(const std::string& k) const {
    NCG_CHECK(has(k), "missing required argument --{}", k);
    return kv_.at(k);
  }
  int get_int(const std::string& k, int def) const {
    return has(k) ? std::stoi(kv_.at(k)) : def;
  }
  float get_float(const std::string& k, float def) const {
    return has(k) ? std::stof(kv_.at(k)) : def;
  }

private:
  std::unordered_map<std::string, std::string> kv_;
};

}  // namespace ncg::app

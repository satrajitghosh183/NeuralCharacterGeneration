#pragma once

#include <ncg/core/tensor.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

namespace ncg::record {

/// Records artifacts and metrics for one pipeline run into a structured directory:
///   <root>/<name>[_<timestamp>]/
///     config/config.json        (run config snapshot)
///     metrics.jsonl             (one JSON object per logged scalar/text/timing)
///     images/<stage>__<name>.png
///     checkpoints/              (model/gaussian dumps)
/// Used at every pipeline stage so runs are reproducible and comparable. Reusable across
/// all phases (selection, fitting, reconstruction, relighting, eval).
class Recorder {
public:
  /// Creates the run directory tree. With `unique_suffix`, appends a local timestamp so
  /// repeated runs don't collide.
  static Recorder create(const std::string& root, const std::string& name,
                         bool unique_suffix = true);

  const std::filesystem::path& dir() const { return dir_; }

  void set_config(const std::string& json_text);
  void log_scalar(const std::string& stage, const std::string& key, double value);
  void log_text(const std::string& stage, const std::string& key, const std::string& text);
  void log_image(const std::string& stage, const std::string& name, const Tensor& image_chw);

  /// Path under checkpoints/ for the caller to write to (creates nothing).
  std::filesystem::path checkpoint_path(const std::string& filename) const;

  /// RAII timer: logs "<key>_ms" for the enclosing scope on destruction.
  class ScopedTimer {
  public:
    ScopedTimer(Recorder* rec, std::string stage, std::string key);
    ~ScopedTimer();
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = default;

  private:
    Recorder* rec_;
    std::string stage_;
    std::string key_;
    std::chrono::steady_clock::time_point t0_;
  };

  ScopedTimer time(const std::string& stage, const std::string& key) {
    return ScopedTimer(this, stage, key);
  }

private:
  explicit Recorder(std::filesystem::path dir);
  void write_line(const std::string& stage, const std::string& key, const char* kind, double value,
                  const std::string& text);

  std::filesystem::path dir_;
  std::shared_ptr<std::ofstream> metrics_;  // shared so Recorder stays movable
  int64_t step_ = 0;
};

}  // namespace ncg::record

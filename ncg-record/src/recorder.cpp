#include <ncg/record/recorder.hpp>

#include <ncg/core/error.hpp>
#include <ncg/core/logging.hpp>
#include <ncg/io/image.hpp>

#include <nlohmann/json.hpp>

#include <ctime>

namespace ncg::record {
namespace {

std::string timestamp() {
  const std::time_t now = std::time(nullptr);
  std::tm tm_buf{};
#if defined(_WIN32)
  localtime_s(&tm_buf, &now);
#else
  localtime_r(&now, &tm_buf);
#endif
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm_buf);
  return buf;
}

std::string sanitize(const std::string& s) {
  std::string out = s;
  for (char& ch : out) {
    if (ch == '/' || ch == ' ' || ch == ':') ch = '_';
  }
  return out;
}

}  // namespace

Recorder::Recorder(std::filesystem::path dir) : dir_(std::move(dir)) {}

Recorder Recorder::create(const std::string& root, const std::string& name, bool unique_suffix) {
  std::filesystem::path dir =
      std::filesystem::path(root) / (unique_suffix ? name + "_" + timestamp() : name);
  std::filesystem::create_directories(dir / "images");
  std::filesystem::create_directories(dir / "checkpoints");
  std::filesystem::create_directories(dir / "config");

  Recorder rec(dir);
  rec.metrics_ = std::make_shared<std::ofstream>(dir / "metrics.jsonl", std::ios::app);
  NCG_CHECK(rec.metrics_->good(), "Recorder: cannot open metrics.jsonl in '{}'", dir.string());
  NCG_LOG_INFO("recording run -> {}", dir.string());
  return rec;
}

void Recorder::write_line(const std::string& stage, const std::string& key, const char* kind,
                          double value, const std::string& text) {
  nlohmann::json j;
  j["step"] = step_++;
  j["stage"] = stage;
  j["key"] = key;
  if (std::string(kind) == "text") {
    j["text"] = text;
  } else {
    j["value"] = value;
  }
  (*metrics_) << j.dump() << '\n';
  metrics_->flush();
}

void Recorder::set_config(const std::string& json_text) {
  std::ofstream os(dir_ / "config" / "config.json");
  NCG_CHECK(os.good(), "Recorder: cannot write config.json");
  os << json_text;
}

void Recorder::log_scalar(const std::string& stage, const std::string& key, double value) {
  write_line(stage, key, "scalar", value, "");
}

void Recorder::log_text(const std::string& stage, const std::string& key, const std::string& text) {
  write_line(stage, key, "text", 0.0, text);
}

void Recorder::log_image(const std::string& stage, const std::string& name,
                         const Tensor& image_chw) {
  const auto path = dir_ / "images" / (sanitize(stage) + "__" + sanitize(name) + ".png");
  io::save_png(path.string(), image_chw);
  log_text(stage, "image:" + name, path.filename().string());
}

std::filesystem::path Recorder::checkpoint_path(const std::string& filename) const {
  return dir_ / "checkpoints" / filename;
}

Recorder::ScopedTimer::ScopedTimer(Recorder* rec, std::string stage, std::string key)
    : rec_(rec), stage_(std::move(stage)), key_(std::move(key)),
      t0_(std::chrono::steady_clock::now()) {}

Recorder::ScopedTimer::~ScopedTimer() {
  if (rec_ == nullptr) return;
  const auto dt = std::chrono::steady_clock::now() - t0_;
  const double ms = std::chrono::duration<double, std::milli>(dt).count();
  rec_->log_scalar(stage_, key_ + "_ms", ms);
}

}  // namespace ncg::record

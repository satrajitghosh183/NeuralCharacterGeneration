#include <ncg/core/logging.hpp>

#include <spdlog/sinks/stdout_color_sinks.h>

#include <cstdlib>
#include <mutex>
#include <string_view>

namespace ncg {
namespace {

constexpr const char* kLoggerName = "ncg";

spdlog::level::level_enum level_from_env() {
  const char* env = std::getenv("NCG_LOG_LEVEL");
  if (env == nullptr) return spdlog::level::info;
  const std::string_view v{env};
  if (v == "trace") return spdlog::level::trace;
  if (v == "debug") return spdlog::level::debug;
  if (v == "info") return spdlog::level::info;
  if (v == "warn") return spdlog::level::warn;
  if (v == "error") return spdlog::level::err;
  if (v == "off") return spdlog::level::off;
  return spdlog::level::info;
}

}  // namespace

void init_logging() {
  static std::once_flag once;
  std::call_once(once, [] {
    auto lg = spdlog::stdout_color_mt(kLoggerName);
    lg->set_level(level_from_env());
    lg->set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
  });
}

std::shared_ptr<spdlog::logger> logger() {
  auto lg = spdlog::get(kLoggerName);
  if (!lg) {
    init_logging();
    lg = spdlog::get(kLoggerName);
  }
  return lg;
}

}  // namespace ncg

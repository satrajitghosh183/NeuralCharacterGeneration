#pragma once

#include <spdlog/spdlog.h>

#include <memory>

namespace ncg {

/// Initializes the global "ncg" logger. Idempotent. Reads level from env NCG_LOG_LEVEL
/// (trace|debug|info|warn|error|off), defaulting to info. Safe to call multiple times.
void init_logging();

/// The shared "ncg" logger. init_logging() is invoked lazily on first use.
std::shared_ptr<spdlog::logger> logger();

}  // namespace ncg

// Thin macros so the logger pointer is fetched once per call site. Use fmt syntax.
#define NCG_LOG_TRACE(...) ::ncg::logger()->trace(__VA_ARGS__)
#define NCG_LOG_DEBUG(...) ::ncg::logger()->debug(__VA_ARGS__)
#define NCG_LOG_INFO(...) ::ncg::logger()->info(__VA_ARGS__)
#define NCG_LOG_WARN(...) ::ncg::logger()->warn(__VA_ARGS__)
#define NCG_LOG_ERROR(...) ::ncg::logger()->error(__VA_ARGS__)

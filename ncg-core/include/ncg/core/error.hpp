#pragma once

#include <spdlog/fmt/fmt.h>

#include <stdexcept>
#include <string>

namespace ncg {

/// Exception type carrying the originating source location.
class NcgError : public std::runtime_error {
public:
  NcgError(std::string msg, const char* file, int line)
      : std::runtime_error(std::move(msg)), file_(file), line_(line) {}

  const char* file() const noexcept { return file_; }
  int line() const noexcept { return line_; }

private:
  const char* file_;
  int line_;
};

namespace detail {
/// Logs at error level and throws NcgError. Defined in error.cpp.
[[noreturn]] void throw_error(std::string msg, const char* file, int line);
}  // namespace detail

}  // namespace ncg

/// Throw an NcgError with an fmt-formatted message.
#define NCG_THROW(...) ::ncg::detail::throw_error(::fmt::format(__VA_ARGS__), __FILE__, __LINE__)

/// Throw unless `cond` holds. Remaining args are an fmt message describing the failure.
#define NCG_CHECK(cond, ...)                                                                      \
  do {                                                                                            \
    if (!(cond)) {                                                                                \
      ::ncg::detail::throw_error("Check failed (" #cond "): " + ::fmt::format(__VA_ARGS__),       \
                                 __FILE__, __LINE__);                                             \
    }                                                                                             \
  } while (0)

/// Marker for unimplemented stubs (skeleton modules).
#define NCG_NOT_IMPLEMENTED() NCG_THROW("not implemented: {}", __func__)

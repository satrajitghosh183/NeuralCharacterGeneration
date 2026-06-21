#pragma once

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

#if defined(__CUDACC__)
// In CUDA translation units we must NOT include fmt: spdlog's bundled fmt headers do not
// parse under nvcc. Keep the format string and ignore the args (these are host-side checks
// in kernel wrappers, so the placeholders simply remain literal in the message).
#include <utility>
namespace ncg::detail {
template <typename... Ts>
inline std::string cuda_msg(const char* fmt, const Ts&...) {
  return std::string(fmt);
}
}  // namespace ncg::detail

#define NCG_THROW(...) ::ncg::detail::throw_error(::ncg::detail::cuda_msg(__VA_ARGS__), __FILE__, __LINE__)
#define NCG_CHECK(cond, ...)                                                                      \
  do {                                                                                            \
    if (!(cond)) {                                                                                \
      ::ncg::detail::throw_error("Check failed (" #cond "): " + ::ncg::detail::cuda_msg(__VA_ARGS__), \
                                 __FILE__, __LINE__);                                             \
    }                                                                                             \
  } while (0)
#define NCG_NOT_IMPLEMENTED() ::ncg::detail::throw_error("not implemented", __FILE__, __LINE__)

#else  // host C++: full fmt formatting.
#include <spdlog/fmt/fmt.h>

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

#endif  // __CUDACC__

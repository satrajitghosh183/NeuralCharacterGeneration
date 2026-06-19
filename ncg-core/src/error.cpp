#include <ncg/core/error.hpp>

#include <ncg/core/logging.hpp>

namespace ncg::detail {

void throw_error(std::string msg, const char* file, int line) {
  NCG_LOG_ERROR("{}:{}: {}", file, line, msg);
  throw NcgError(std::move(msg), file, line);
}

}  // namespace ncg::detail

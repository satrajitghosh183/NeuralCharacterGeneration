#pragma once

#include <ncg/core/config.hpp>

#ifdef NCG_WITH_CUDA

#include <ncg/core/error.hpp>

#include <cuda_runtime_api.h>

namespace ncg::detail {

[[noreturn]] inline void cuda_fail(const char* expr, cudaError_t err, const char* file, int line) {
  throw_error(::fmt::format("CUDA error {} ({}) at: {}", static_cast<int>(err),
                            cudaGetErrorString(err), expr),
              file, line);
}

/// Whether NCG_CUDA_KERNEL_CHECK() should cudaDeviceSynchronize() after each launch.
/// Defaults: on in Debug, off in Release. Override with env NCG_CUDA_SYNC=0|1.
/// Defined in device.cpp.
bool kernel_sync_check_enabled();

}  // namespace ncg::detail

/// Check a CUDA runtime call's return code.
#define NCG_CUDA_CHECK(expr)                                                                       \
  do {                                                                                             \
    cudaError_t _ncg_e = (expr);                                                                   \
    if (_ncg_e != cudaSuccess) ::ncg::detail::cuda_fail(#expr, _ncg_e, __FILE__, __LINE__);        \
  } while (0)

/// Call immediately after a kernel launch: checks the launch error, and (when enabled)
/// synchronizes to surface otherwise-async faults at this exact site.
#define NCG_CUDA_KERNEL_CHECK()                                                                    \
  do {                                                                                             \
    NCG_CUDA_CHECK(cudaGetLastError());                                                            \
    if (::ncg::detail::kernel_sync_check_enabled()) NCG_CUDA_CHECK(cudaDeviceSynchronize());       \
  } while (0)

#endif  // NCG_WITH_CUDA

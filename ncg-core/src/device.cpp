#include <ncg/core/device.hpp>

#include <ncg/core/config.hpp>

#ifdef NCG_WITH_CUDA
#include <ncg/core/cuda_check.hpp>

#include <cstdlib>
#include <string_view>
#endif

namespace ncg {

bool cuda_available() { return torch::cuda::is_available(); }

at::Device default_device() {
  return cuda_available() ? at::Device(at::kCUDA, 0) : at::Device(at::kCPU);
}

void set_deterministic_fp32(bool enable) {
  // When enabling determinism we forbid TF32 (Hopper uses it for fp32 matmul by default).
  auto& ctx = at::globalContext();
  ctx.setAllowTF32CuBLAS(!enable);
  ctx.setAllowTF32CuDNN(!enable);
}

#ifdef NCG_WITH_CUDA
namespace detail {

bool kernel_sync_check_enabled() {
  static const bool value = [] {
    if (const char* env = std::getenv("NCG_CUDA_SYNC")) {
      return std::string_view{env} != "0";
    }
#ifdef NDEBUG
    return false;  // Release: off (no per-launch sync overhead).
#else
    return true;  // Debug: on (surface async faults at the launch site).
#endif
  }();
  return value;
}

}  // namespace detail
#endif  // NCG_WITH_CUDA

}  // namespace ncg

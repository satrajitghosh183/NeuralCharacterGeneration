#include <ncg/core/device.hpp>

#include <ncg/core/config.hpp>

#include <cstdlib>
#include <string_view>

#ifdef NCG_WITH_CUDA
#include <ncg/core/cuda_check.hpp>
#endif

namespace ncg {

bool cuda_available() { return torch::cuda::is_available(); }

bool mps_available() {
#ifdef __APPLE__
  return at::hasMPS();
#else
  return false;
#endif
}

at::Device default_device() {
  return cuda_available() ? at::Device(at::kCUDA, 0) : at::Device(at::kCPU);
}

at::Device compute_device() {
  if (const char* env = std::getenv("NCG_DEVICE")) {
    const std::string_view v{env};
    if (v == "cpu") return at::Device(at::kCPU);
    if (v == "cuda" && cuda_available()) return at::Device(at::kCUDA, 0);
    if (v == "mps" && mps_available()) return at::Device(at::kMPS);
  }
  if (cuda_available()) return at::Device(at::kCUDA, 0);
  if (mps_available()) return at::Device(at::kMPS);
  return at::Device(at::kCPU);
}

bool gpu_available() { return compute_device().type() != at::kCPU; }

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

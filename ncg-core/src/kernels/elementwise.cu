#include <ncg/core/kernels/elementwise.hpp>

#include <ncg/core/cuda_check.hpp>
#include <ncg/core/error.hpp>
#include <ncg/core/kernel_registry.hpp>

#include <cstdint>

namespace ncg {
namespace {

__global__ void saxpy_kernel(const float* __restrict__ x, const float* __restrict__ y, float a,
                             float* __restrict__ out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a * x[i] + y[i];
}

}  // namespace

Tensor saxpy(const Tensor& x, const Tensor& y, double a) {
  require_cuda_f32_contiguous(x, "saxpy(x)");
  require_cuda_f32_contiguous(y, "saxpy(y)");
  NCG_CHECK(x.sizes() == y.sizes(), "saxpy: shape mismatch ({} vs {})", x.sizes(), y.sizes());

  Tensor out = at::empty_like(x);
  const int64_t n = x.numel();
  if (n == 0) return out;

  constexpr unsigned kBlock = 256;
  const unsigned grid = grid_1d(n, kBlock);
  saxpy_kernel<<<grid, kBlock>>>(x.data_ptr<float>(), y.data_ptr<float>(),
                                 static_cast<float>(a), out.data_ptr<float>(), n);
  NCG_CUDA_KERNEL_CHECK();
  return out;
}

}  // namespace ncg

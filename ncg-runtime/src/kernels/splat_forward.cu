#include <ncg/runtime/splat_raster.hpp>

#include <ncg/core/cuda_check.hpp>
#include <ncg/core/error.hpp>

#include <cstdint>

namespace ncg::runtime {
namespace {

// One thread per pixel. Composites all N Gaussians front-to-back (inputs sorted near->far).
// O(pixels * N) — deliberately simple for the Phase-1 slice; Phase 2 replaces this with a
// tiled rasterizer + backward pass.
__global__ void splat_forward_kernel(const float* __restrict__ u, const float* __restrict__ v,
                                     const float* __restrict__ inv_s2,
                                     const float* __restrict__ opacity,
                                     const float* __restrict__ colors,  // [N,3]
                                     int64_t n, int height, int width, float bg0, float bg1,
                                     float bg2, float* __restrict__ out_img,  // [3,H,W]
                                     float* __restrict__ out_alpha) {         // [H,W]
  const int px = blockIdx.x * blockDim.x + threadIdx.x;
  const int py = blockIdx.y * blockDim.y + threadIdx.y;
  if (px >= width || py >= height) return;

  const float fpx = static_cast<float>(px) + 0.5F;
  const float fpy = static_cast<float>(py) + 0.5F;

  float c0 = 0.0F;
  float c1 = 0.0F;
  float c2 = 0.0F;
  float trans = 1.0F;  // remaining transmittance

  for (int64_t i = 0; i < n; ++i) {
    const float dx = fpx - u[i];
    const float dy = fpy - v[i];
    const float d2 = (dx * dx + dy * dy) * inv_s2[i];
    if (d2 > 9.0F) continue;  // beyond 3 sigma

    float a = opacity[i] * __expf(-0.5F * d2);
    if (a < 1.0F / 255.0F) continue;
    if (a > 0.99F) a = 0.99F;

    const float w = trans * a;
    c0 += w * colors[i * 3 + 0];
    c1 += w * colors[i * 3 + 1];
    c2 += w * colors[i * 3 + 2];
    trans *= (1.0F - a);
    if (trans < 1e-4F) break;
  }

  const int hw = height * width;
  const int idx = py * width + px;
  out_img[0 * hw + idx] = c0 + trans * bg0;
  out_img[1 * hw + idx] = c1 + trans * bg1;
  out_img[2 * hw + idx] = c2 + trans * bg2;
  out_alpha[idx] = 1.0F - trans;
}

}  // namespace

std::pair<Tensor, Tensor> splat_render_cuda(const Tensor& u, const Tensor& v,
                                            const Tensor& inv_sigma2, const Tensor& opacity,
                                            const Tensor& colors, int height, int width,
                                            const std::array<float, 3>& background) {
  require_cuda_f32_contiguous(u, "splat:u");
  require_cuda_f32_contiguous(v, "splat:v");
  require_cuda_f32_contiguous(inv_sigma2, "splat:inv_sigma2");
  require_cuda_f32_contiguous(opacity, "splat:opacity");
  require_cuda_f32_contiguous(colors, "splat:colors");
  NCG_CHECK(colors.dim() == 2 && colors.size(1) == 3, "splat: colors must be [N,3]");
  const int64_t n = u.numel();
  NCG_CHECK(colors.size(0) == n, "splat: colors/N mismatch");

  auto opts = u.options();
  Tensor image = at::zeros({3, height, width}, opts);
  Tensor alpha = at::zeros({1, height, width}, opts);

  const dim3 block(16, 16);
  const dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  splat_forward_kernel<<<grid, block>>>(
      u.data_ptr<float>(), v.data_ptr<float>(), inv_sigma2.data_ptr<float>(),
      opacity.data_ptr<float>(), colors.data_ptr<float>(), n, height, width, background[0],
      background[1], background[2], image.data_ptr<float>(), alpha.data_ptr<float>());
  NCG_CUDA_KERNEL_CHECK();

  return {image, alpha};
}

}  // namespace ncg::runtime

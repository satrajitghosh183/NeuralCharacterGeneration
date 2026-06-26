#include <ncg/recon/uv_texture.hpp>

#include <ncg/core/error.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <cmath>

namespace ncg::recon {

UVRaster uv_rasterize(const Tensor& uv_coords_in, const Tensor& uv_faces_in, int res) {
  NCG_CHECK(res > 0, "uv_rasterize: res must be > 0");
  const auto uv = uv_coords_in.detach().to(at::kCPU, at::kFloat).contiguous();  // [n_uv,2]
  const auto uf = uv_faces_in.detach().to(at::kCPU, at::kLong).contiguous();    // [F,3]
  const int64_t F = uf.size(0);

  auto face = torch::full({static_cast<int64_t>(res) * res}, -1,
                          at::TensorOptions().dtype(at::kLong));
  auto bary = torch::zeros({static_cast<int64_t>(res) * res, 3},
                           at::TensorOptions().dtype(at::kFloat));
  const auto uva = uv.accessor<float, 2>();
  const auto ufa = uf.accessor<int64_t, 2>();
  auto fa = face.accessor<int64_t, 1>();
  auto ba = bary.accessor<float, 2>();

  for (int64_t f = 0; f < F; ++f) {
    const int64_t i0 = ufa[f][0], i1 = ufa[f][1], i2 = ufa[f][2];
    const float fx0 = uva[i0][0] * res, fy0 = uva[i0][1] * res;
    const float fx1 = uva[i1][0] * res, fy1 = uva[i1][1] * res;
    const float fx2 = uva[i2][0] * res, fy2 = uva[i2][1] * res;
    const int minc = std::max(0, static_cast<int>(std::floor(std::min({fx0, fx1, fx2}))));
    const int maxc = std::min(res - 1, static_cast<int>(std::ceil(std::max({fx0, fx1, fx2}))));
    const int minr = std::max(0, static_cast<int>(std::floor(std::min({fy0, fy1, fy2}))));
    const int maxr = std::min(res - 1, static_cast<int>(std::ceil(std::max({fy0, fy1, fy2}))));
    const float denom = (fy1 - fy2) * (fx0 - fx2) + (fx2 - fx1) * (fy0 - fy2);
    if (std::abs(denom) < 1e-12F) continue;
    for (int r = minr; r <= maxr; ++r) {
      for (int c = minc; c <= maxc; ++c) {
        const float px = static_cast<float>(c) + 0.5F, py = static_cast<float>(r) + 0.5F;
        const float b0 = ((fy1 - fy2) * (px - fx2) + (fx2 - fx1) * (py - fy2)) / denom;
        const float b1 = ((fy2 - fy0) * (px - fx2) + (fx0 - fx2) * (py - fy2)) / denom;
        const float b2 = 1.0F - b0 - b1;
        if (b0 >= -1e-4F && b1 >= -1e-4F && b2 >= -1e-4F) {
          const int64_t t = static_cast<int64_t>(r) * res + c;
          fa[t] = f;
          ba[t][0] = b0;
          ba[t][1] = b1;
          ba[t][2] = b2;
        }
      }
    }
  }
  UVRaster out;
  out.face = face;
  out.bary = bary;
  out.res = res;
  return out;
}

Tensor bake_to_uv(const UVRaster& ras, const Tensor& vertex_values, const Tensor& faces,
                  Tensor& mask_out) {
  const int res = ras.res;
  const auto vv = vertex_values.detach().to(at::kCPU, at::kFloat).contiguous();  // [V,C]
  const auto fc = faces.detach().to(at::kCPU, at::kLong).contiguous();           // [F,3]
  const int64_t C = vv.size(1);
  auto tex = torch::zeros({static_cast<int64_t>(res) * res, C}, at::kFloat);
  auto mask = torch::zeros({static_cast<int64_t>(res) * res}, at::kFloat);
  const auto face_cpu = ras.face.to(at::kCPU).contiguous();
  const auto bary_cpu = ras.bary.to(at::kCPU).contiguous();
  const auto faceacc = face_cpu.accessor<int64_t, 1>();
  const auto baryacc = bary_cpu.accessor<float, 2>();
  const auto vva = vv.accessor<float, 2>();
  const auto fca = fc.accessor<int64_t, 2>();
  auto ta = tex.accessor<float, 2>();
  auto ma = mask.accessor<float, 1>();

  for (int64_t t = 0; t < static_cast<int64_t>(res) * res; ++t) {
    const int64_t f = faceacc[t];
    if (f < 0) continue;
    const int64_t a = fca[f][0], b = fca[f][1], c = fca[f][2];
    const float w0 = baryacc[t][0], w1 = baryacc[t][1], w2 = baryacc[t][2];
    for (int64_t ch = 0; ch < C; ++ch) {
      ta[t][ch] = w0 * vva[a][ch] + w1 * vva[b][ch] + w2 * vva[c][ch];
    }
    ma[t] = 1.0F;
  }
  mask_out = mask.view({res, res});
  return tex.view({res, res, C});
}

}  // namespace ncg::recon

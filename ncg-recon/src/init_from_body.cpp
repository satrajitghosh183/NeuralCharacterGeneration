#include <ncg/recon/init_from_body.hpp>

#include <ncg/core/error.hpp>

#include <torch/torch.h>

#include <algorithm>
#include <limits>

namespace ncg::recon {

Tensor per_vertex_scale(const Tensor& vertices, double mult, int k) {
  NCG_CHECK(vertices.dim() == 2 && vertices.size(1) == 3,
            "per_vertex_scale: vertices must be [V,3]");
  const auto v = vertices.detach().to(at::kFloat).contiguous();
  const int64_t n = v.size(0);
  NCG_CHECK(n >= 2, "per_vertex_scale: need at least 2 vertices");
  const int64_t kk = std::min<int64_t>(std::max(1, k), n - 1);

  auto dist = torch::cdist(v, v);  // [n,n] pairwise (one-time init cost)
  dist.diagonal().fill_(std::numeric_limits<float>::infinity());  // exclude self
  const auto nn = std::get<0>(dist.topk(kk, /*dim=*/1, /*largest=*/false)).mean(1);  // [n]
  return (nn * mult).clamp_min(1e-6);
}

GaussianCloud gaussians_on_body(const Tensor& vertices, float scale, const Tensor& colors,
                                const Tensor& per_vertex_scale_t) {
  NCG_CHECK(vertices.dim() == 2 && vertices.size(1) == 3, "gaussians_on_body: vertices must be [V,3]");

  const auto verts = vertices.detach().to(at::kFloat).contiguous();
  const int64_t n = verts.size(0);
  const auto opts = verts.options();

  GaussianCloud g;
  g.positions = verts;
  if (per_vertex_scale_t.defined()) {
    NCG_CHECK(per_vertex_scale_t.dim() == 1 && per_vertex_scale_t.size(0) == n,
              "gaussians_on_body: per_vertex_scale must be [V] matching vertices");
    g.scales = per_vertex_scale_t.detach().to(opts).view({n, 1}).expand({n, 3}).contiguous();
  } else {
    NCG_CHECK(scale > 0.0F, "gaussians_on_body: scale must be positive");
    g.scales = torch::full({n, 3}, scale, opts);
  }

  // Identity quaternion (w,x,y,z) = (1,0,0,0).
  g.rotations = torch::zeros({n, 4}, opts);
  g.rotations.select(1, 0).fill_(1.0);

  g.opacities = torch::ones({n, 1}, opts);

  if (colors.defined()) {
    NCG_CHECK(colors.dim() == 2 && colors.size(0) == n && colors.size(1) == 3,
              "gaussians_on_body: colors must be [V,3] matching vertices");
    g.colors = colors.detach().to(opts).contiguous();
  } else {
    g.colors = torch::full({n, 3}, 0.6F, opts);  // neutral gray
  }

  g.validate();
  return g;
}

}  // namespace ncg::recon

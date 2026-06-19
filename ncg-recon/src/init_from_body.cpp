#include <ncg/recon/init_from_body.hpp>

#include <ncg/core/error.hpp>

namespace ncg::recon {

GaussianCloud gaussians_on_body(const Tensor& vertices, float scale, const Tensor& colors) {
  NCG_CHECK(vertices.dim() == 2 && vertices.size(1) == 3, "gaussians_on_body: vertices must be [V,3]");
  NCG_CHECK(scale > 0.0F, "gaussians_on_body: scale must be positive");

  const auto verts = vertices.detach().to(at::kFloat).contiguous();
  const int64_t n = verts.size(0);
  const auto opts = verts.options();

  GaussianCloud g;
  g.positions = verts;
  g.scales = torch::full({n, 3}, scale, opts);

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

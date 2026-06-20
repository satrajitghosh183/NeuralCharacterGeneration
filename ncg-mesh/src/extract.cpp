#include <ncg/mesh/extract.hpp>

#include <ncg/core/error.hpp>

#include <algorithm>

namespace ncg::mesh {

TriMesh extract_mesh(const recon::GaussianCloud& g, int res) {
  g.validate();
  NCG_CHECK(res >= 4, "extract_mesh: grid_resolution must be >= 4");

  const auto dev = g.positions.device();
  const auto opts = at::TensorOptions().dtype(at::kFloat).device(dev);
  const auto pos = g.positions.detach().to(at::kFloat);          // [N,3]
  const auto sig = g.scales.mean(1).to(at::kFloat).clamp_min(1e-4);  // [N]
  const auto op = g.opacities.squeeze(1).to(at::kFloat);         // [N]

  const auto mn = std::get<0>(pos.min(/*dim=*/0));  // [3]
  const auto mx = std::get<0>(pos.max(/*dim=*/0));  // [3]
  const float pad = 3.0F * sig.max().item<float>() + 1e-3F;

  const auto lo = (mn - pad).to(at::kCPU);
  const auto hi = (mx + pad).to(at::kCPU);
  std::array<float, 3> origin{lo[0].item<float>(), lo[1].item<float>(), lo[2].item<float>()};
  std::array<float, 3> spacing{};
  for (int k = 0; k < 3; ++k) {
    spacing[static_cast<size_t>(k)] =
        (hi[k].item<float>() - origin[static_cast<size_t>(k)]) / static_cast<float>(res - 1);
  }

  // Grid points ordered with x fastest, matching marching_cubes' field indexing.
  const auto xs = torch::linspace(origin[0], hi[0].item<float>(), res, opts);
  const auto ys = torch::linspace(origin[1], hi[1].item<float>(), res, opts);
  const auto zs = torch::linspace(origin[2], hi[2].item<float>(), res, opts);
  const auto gx = xs.view({1, 1, res}).expand({res, res, res});
  const auto gy = ys.view({1, res, 1}).expand({res, res, res});
  const auto gz = zs.view({res, 1, 1}).expand({res, res, res});
  const auto points = torch::stack({gx, gy, gz}, -1).reshape({-1, 3});  // [res^3,3]

  const int64_t total = points.size(0);
  std::vector<float> field(static_cast<size_t>(total));
  const auto inv_var = (1.0 / (sig * sig)).view({1, -1});  // [1,N]
  const auto op_row = op.view({1, -1});                    // [1,N]

  const int64_t chunk = 8192;
  for (int64_t s = 0; s < total; s += chunk) {
    const int64_t e = std::min(s + chunk, total);
    const auto pc = points.index({torch::indexing::Slice(s, e)});       // [C,3]
    const auto d2 = torch::cdist(pc, pos).pow(2);                       // [C,N]
    const auto w = op_row * torch::exp(-0.5 * d2 * inv_var);            // [C,N]
    const auto fc = w.sum(1).to(at::kCPU, at::kFloat).contiguous();     // [C]
    std::copy(fc.data_ptr<float>(), fc.data_ptr<float>() + (e - s),
              field.begin() + static_cast<std::ptrdiff_t>(s));
  }

  // Isolevel: a fraction of the peak density (heuristic; tune per use).
  float fmax = 0.0F;
  for (float v : field) fmax = std::max(fmax, v);
  const float iso = std::max(1e-3F, 0.25F * fmax);

  return marching_cubes(field, res, res, res, iso, origin, spacing);
}

}  // namespace ncg::mesh

#include <ncg/fit/splat_bind.hpp>

#include <ncg/core/error.hpp>

#include <torch/torch.h>

#include <tuple>

namespace ncg::fit {

SplatBinding bind_splats_knn(const Tensor& centers_in, const Tensor& verts_in, int64_t k,
                             float h_scale) {
  const auto centers = centers_in.to(at::kCPU, at::kFloat).contiguous();  // [N,3]
  const auto verts = verts_in.to(at::kCPU, at::kFloat).contiguous();      // [V,3]
  const int64_t kk = std::min<int64_t>(k, verts.size(0));
  const auto d = torch::cdist(centers, verts);                            // [N,V]
  const auto topk = torch::topk(d, kk, /*dim=*/1, /*largest=*/false);     // (dist[N,k], idx[N,k])
  const auto dist = std::get<0>(topk), idx = std::get<1>(topk);
  const auto h = (h_scale * dist.mean(1, /*keepdim=*/true)).clamp_min(1e-6F);  // [N,1] bandwidth
  auto w = torch::exp(-(dist * dist) / (h * h));                          // [N,k]
  w = w / w.sum(1, true).clamp_min(1e-12);
  SplatBinding b;
  b.idx = idx.contiguous();
  b.weight = w.contiguous();
  return b;
}

Tensor blend_vertex_transforms(const Tensor& vt_in, const SplatBinding& binding) {
  const auto vt = vt_in.to(at::kCPU, at::kFloat).contiguous();            // [V,4,4]
  const auto idx = binding.idx, w = binding.weight;                      // [N,k]
  const int64_t N = idx.size(0), k = idx.size(1);
  const auto vt_knn = vt.index_select(0, idx.reshape({-1})).reshape({N, k, 4, 4});
  return (vt_knn * w.reshape({N, k, 1, 1})).sum(1);                       // [N,4,4] (LBS linear blend)
}

double swim_metric(const Tensor& centers_in, const Tensor& /*verts_in*/, const Tensor& vt_in,
                   const SplatBinding& binding) {
  const auto centers = centers_in.to(at::kCPU, at::kFloat).contiguous();  // [N,3]
  const auto vt = vt_in.to(at::kCPU, at::kFloat).contiguous();            // [V,4,4]
  const auto idx = binding.idx, w = binding.weight;
  const int64_t N = idx.size(0), k = idx.size(1);

  // μ'_s = (Σ_j ω_j T_j) · μ_s  — blend transforms, THEN apply (what deform_avatar does, LBS).
  const auto B = blend_vertex_transforms(vt, binding);                    // [N,4,4]
  const auto Rb = B.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3),
                           torch::indexing::Slice(0, 3)});
  const auto tb = B.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 3});
  const auto mu = torch::einsum("nab,nb->na", {Rb, centers}) + tb;        // [N,3]

  // Φ_s = Σ_j ω_j (T_j · μ_s) — apply each transform to the SAME splat point, THEN blend. The gap
  // μ'−Φ is the pure LBS-linearization error on the splat — i.e. the swim (0 for rigid / 1:1; small
  // when the bound verts' transforms vary smoothly). Both use μ_s, so a splat's legitimate offset-
  // following cancels out and only true detachment remains.
  const auto vt_knn = vt.index_select(0, idx.reshape({-1})).reshape({N, k, 4, 4});
  const auto Rk = vt_knn.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                torch::indexing::Slice(0, 3), torch::indexing::Slice(0, 3)});
  const auto tk = vt_knn.index({torch::indexing::Slice(), torch::indexing::Slice(),
                                torch::indexing::Slice(0, 3), 3});        // [N,k,3]
  const auto posed = torch::einsum("nkab,nb->nka", {Rk, centers}) + tk;   // [N,k,3] T_j·μ_s
  const auto Phi = (posed * w.unsqueeze(2)).sum(1);                       // [N,3]

  return (mu - Phi).norm(2, 1).mean().item<double>();
}

}  // namespace ncg::fit

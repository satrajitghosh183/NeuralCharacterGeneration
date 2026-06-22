#include <ncg/recon/inverse_render.hpp>

#include <ncg/core/error.hpp>

#include <torch/torch.h>

#include <vector>

namespace ncg::recon {

Tensor sh_basis(const Tensor& normals_in) {
  NCG_CHECK(normals_in.size(-1) == 3, "sh_basis: normals last dim must be 3");
  const auto n = normals_in / normals_in.norm(2, -1, /*keepdim=*/true).clamp_min(1e-8);
  const auto x = n.select(-1, 0);
  const auto y = n.select(-1, 1);
  const auto z = n.select(-1, 2);
  const auto one = torch::ones_like(x);
  // Real order-2 SH (Y_lm) with the Lambertian half-cosine convolution A_l folded in
  // (A_0=pi, A_1=2pi/3, A_2=pi/4), so irradiance E(n) = L . sh_basis(n).
  std::vector<Tensor> c = {
      0.886227F * one,                                            // l=0
      1.023328F * y, 1.023328F * z, 1.023328F * x,                // l=1
      0.858086F * (x * y), 0.858086F * (y * z),                   // l=2
      0.247708F * (3.0F * z * z - 1.0F), 0.858086F * (x * z), 0.429043F * (x * x - y * y)};
  return torch::stack(c, -1);  // [...,9]
}

Tensor shade_sh(const Tensor& albedo, const Tensor& sh, const Tensor& normals) {
  NCG_CHECK(sh.dim() == 2 && sh.size(0) == 3 && sh.size(1) == 9, "shade_sh: sh must be [3,9]");
  const auto b = sh_basis(normals);              // [V,9]
  const auto E = torch::matmul(b, sh.t());       // [V,3] irradiance per channel
  return albedo * E;
}

InverseRenderResult solve_inverse_render(const Tensor& obs, const Tensor& normals,
                                         const Tensor& weights, const InverseRenderConfig& cfg) {
  NCG_CHECK(obs.dim() == 3 && obs.size(2) == 3, "solve_inverse_render: obs must be [N,V,3]");
  NCG_CHECK(normals.sizes() == obs.sizes(), "solve_inverse_render: normals must match obs [N,V,3]");
  NCG_CHECK(weights.dim() == 2 && weights.size(0) == obs.size(0) && weights.size(1) == obs.size(1),
            "solve_inverse_render: weights must be [N,V]");

  const auto opts = obs.options();
  const auto b = sh_basis(normals);                 // [N,V,9]
  const auto w = weights.unsqueeze(-1);             // [N,V,1]
  const auto eye9 = torch::eye(9, opts);

  // Init albedo: visibility-weighted mean of observations (the naive-fuse warm start).
  auto albedo = ((w * obs).sum(0) / w.sum(0).clamp_min(1e-6)).clamp(0.0, 1.5);  // [V,3]
  Tensor lights = torch::zeros({obs.size(0), 3, 9}, opts);
  Tensor precision = torch::ones_like(albedo);

  for (int it = 0; it < cfg.iterations; ++it) {
    // L-step: per photo i, per channel c, weighted 9x9 SH normal equations (closed form).
    const auto bb = b.unsqueeze(-1) * b.unsqueeze(-2);            // [N,V,9,9]
    const auto wa2 = weights.unsqueeze(-1) * albedo.pow(2).unsqueeze(0);  // [N,V,3]
    auto M = torch::einsum("nvc,nvjk->ncjk", {wa2, bb});         // [N,3,9,9]
    M = M + cfg.light_ridge * eye9;
    const auto war = weights.unsqueeze(-1) * albedo.unsqueeze(0) * obs;   // [N,V,3]
    const auto rhs = torch::einsum("nvc,nvk->nck", {war, b});    // [N,3,9]
    lights = torch::linalg_solve(M, rhs.unsqueeze(-1)).squeeze(-1);       // [N,3,9]

    // A-step: per vertex v, per channel c, closed-form scalar solve (parallel over V).
    const auto s = torch::einsum("nck,nvk->nvc", {lights, b});   // [N,V,3] shading
    const auto ws = weights.unsqueeze(-1) * s;                   // [N,V,3]
    const auto num = (ws * obs).sum(0);                          // [V,3]
    const auto den = (ws * s).sum(0) + cfg.albedo_ridge;         // [V,3] = GN precision
    albedo = (num / den).clamp(0.0, 1.5);
    precision = den;
  }

  return {albedo, lights, precision};
}

}  // namespace ncg::recon

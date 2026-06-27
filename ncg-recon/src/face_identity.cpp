#include <ncg/recon/face_identity.hpp>

#include <ncg/core/error.hpp>

#include <torch/torch.h>

#include <cmath>
#include <vector>

namespace ncg::recon {
namespace {

// Ridge least squares: argmin_x ||A x - b||^2 + lam ||x||^2  via the normal equations.
// A [m,n]; b [m] or [m,k]; returns [n] or [n,k].
Tensor ridge_solve(const Tensor& A, const Tensor& b, double lam) {
  const auto n = A.size(1);
  const auto AtA = torch::matmul(A.t(), A) + lam * torch::eye(n, A.options());
  const bool vec = b.dim() == 1;
  const auto Atb = torch::matmul(A.t(), vec ? b.unsqueeze(1) : b);
  const auto x = torch::linalg_solve(AtA, Atb);
  return vec ? x.squeeze(1) : x;
}

}  // namespace

FaceIdentityResult solve_face_identity(const Tensor& base_in, const Tensor& idb_in,
                                       const Tensor& exb_in, const Tensor& lm_in,
                                       const FaceIdentityConfig& cfg) {
  const auto fopt = at::TensorOptions().dtype(at::kFloat);
  const auto base = base_in.to(at::kCPU, at::kFloat).contiguous();  // [L,3]
  const auto idb = idb_in.to(at::kCPU, at::kFloat).contiguous();    // [L,3,nid]
  const auto exb = exb_in.to(at::kCPU, at::kFloat).contiguous();    // [L,3,nex]
  const auto lm = lm_in.to(at::kCPU, at::kFloat).contiguous();      // [N,L,2]
  const int64_t L = base.size(0), nid = idb.size(2), nex = exb.size(2), N = lm.size(0);
  NCG_CHECK(idb.size(0) == L && exb.size(0) == L && lm.size(1) == L, "face_identity: L mismatch");

  auto beta = torch::zeros({nid}, fopt);
  auto psi = torch::zeros({N, nex}, fopt);
  std::vector<Tensor> M(static_cast<size_t>(N)), t(static_cast<size_t>(N));  // M_i [2,3], t_i [2]
  auto w = torch::ones({N}, fopt);

  // Posed-shape landmarks for given identity β and expression ψ.
  auto shape_of = [&](const Tensor& b, const Tensor& p) {
    return base + torch::einsum("lck,k->lc", {idb, b}) + torch::einsum("lck,k->lc", {exb, p});
  };
  // Weak-perspective (affine) projection fit: solve [M|t] (2x4) for u = q·Mᵀ + t.
  auto fit_pose = [&](const Tensor& q, const Tensor& u, Tensor& Mi, Tensor& ti) {
    const auto D = torch::cat({q, torch::ones({L, 1}, fopt)}, 1);  // [L,4]
    const auto Mt = ridge_solve(D, u, 1e-6);                       // [4,2]
    Mi = Mt.slice(0, 0, 3).t().contiguous();                      // [2,3]
    ti = Mt.slice(0, 3, 4).squeeze(0).contiguous();              // [2]
  };

  for (int64_t i = 0; i < N; ++i) fit_pose(base, lm[i], M[static_cast<size_t>(i)], t[static_cast<size_t>(i)]);

  double resid = 0;
  for (int it = 0; it < cfg.iterations; ++it) {
    // ---- per-photo: pose then expression (both linear given the other) ----
    for (int64_t i = 0; i < N; ++i) {
      const auto si = static_cast<size_t>(i);
      fit_pose(shape_of(beta, psi[i]), lm[i], M[si], t[si]);
      // ψ given pose, β:  lm - (M(base+idb·β)ᵀ + t) = M (exb·ψ)ᵀ  (linear in ψ)
      const auto sid = base + torch::einsum("lck,k->lc", {idb, beta});          // [L,3]
      const auto pred0 = torch::matmul(sid, M[si].t()) + t[si].unsqueeze(0);    // [L,2]
      const auto r = (lm[i] - pred0).reshape({L * 2});                          // [2L]
      const auto E = torch::einsum("ab,lbk->lak", {M[si], exb}).reshape({L * 2, nex});  // [2L,nex]
      psi[i] = ridge_solve(E, r, cfg.expr_ridge);
    }
    // ---- shared β: stack every photo's landmark residual, weighted by consistency ----
    std::vector<Tensor> As, bs;
    for (int64_t i = 0; i < N; ++i) {
      const auto si = static_cast<size_t>(i);
      const auto sex = base + torch::einsum("lck,k->lc", {exb, psi[i]});        // [L,3]
      const auto pred0 = torch::matmul(sex, M[si].t()) + t[si].unsqueeze(0);    // [L,2]
      const auto r = (lm[i] - pred0).reshape({L * 2});                          // [2L]
      const auto A = torch::einsum("ab,lbk->lak", {M[si], idb}).reshape({L * 2, nid});  // [2L,nid]
      const float wi = std::sqrt(std::max(1e-4F, w[i].item<float>()));
      As.push_back(A * wi);
      bs.push_back(r * wi);
    }
    beta = ridge_solve(torch::cat(As, 0), torch::cat(bs, 0), cfg.id_ridge);
    // ---- robust per-photo consistency (Welsch on reprojection RMS) ----
    auto rms = torch::zeros({N}, fopt);
    for (int64_t i = 0; i < N; ++i) {
      const auto si = static_cast<size_t>(i);
      const auto pred = torch::matmul(shape_of(beta, psi[i]), M[si].t()) + t[si].unsqueeze(0);
      rms[i] = (lm[i] - pred).pow(2).sum(1).sqrt().mean();
    }
    resid = rms.mean().item<double>();
    if (cfg.robust) {
      const auto c = (cfg.robust_k * rms.median()).clamp_min(1e-6);
      w = torch::exp(-0.5 * (rms / c).pow(2));
    }
  }

  FaceIdentityResult R;
  R.id_shape = beta;
  R.expr = psi;
  R.weight = w;
  R.residual = resid;
  R.scale = torch::zeros({N}, fopt);
  R.trans = torch::zeros({N, 2}, fopt);
  R.rot = torch::zeros({N, 3, 3}, fopt);
  for (int64_t i = 0; i < N; ++i) {
    const auto si = static_cast<size_t>(i);
    R.scale[i] = M[si].norm(2, 1).mean();
    R.trans[i] = t[si];
    const auto r0 = M[si][0] / M[si][0].norm().clamp_min(1e-6);
    auto r1 = M[si][1] - r0 * (r0 * M[si][1]).sum();
    r1 = r1 / r1.norm().clamp_min(1e-6);
    R.rot[i] = torch::stack({r0, r1, torch::cross(r0, r1)}, 0);
  }
  return R;
}

}  // namespace ncg::recon

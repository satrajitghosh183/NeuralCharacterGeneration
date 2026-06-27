#include <ncg/geom/solve_geometry.hpp>

#include <ncg/core/error.hpp>

#include <torch/torch.h>

#include <vector>

namespace ncg::geom {
namespace {

// Ridge least squares (AᵀA+λI)⁻¹Aᵀb. A [m,n]; b [m] or [m,k]; returns [n] or [n,k].
Tensor ridge_solve(const Tensor& A, const Tensor& b, double lam) {
  const auto n = A.size(1);
  const auto AtA = torch::matmul(A.t(), A) + lam * torch::eye(n, A.options());
  const bool vec = b.dim() == 1;
  const auto Atb = torch::matmul(A.t(), vec ? b.unsqueeze(1) : b);
  const auto x = torch::linalg_solve(AtA, Atb);
  return vec ? x.squeeze(1) : x;
}

}  // namespace

Tensor cotangent_laplacian(const Tensor& verts_in, const Tensor& faces_in) {
  const auto v = verts_in.to(at::kCPU, at::kFloat).contiguous();   // [V,3]
  const auto f = faces_in.to(at::kCPU, at::kLong).contiguous();    // [F,3]
  const int64_t V = v.size(0);
  const auto a = f.select(1, 0), b = f.select(1, 1), c = f.select(1, 2);  // [F] vertex ids
  const auto va = v.index_select(0, a), vb = v.index_select(0, b), vc = v.index_select(0, c);
  // cot of the angle at each vertex = (e1·e2)/|e1×e2| for the two incident edges.
  auto cot_at = [](const Tensor& p, const Tensor& q, const Tensor& r) {  // cotangent of angle at p
    const auto e1 = q - p, e2 = r - p;
    const auto cross = torch::cross(e1, e2, 1).norm(2, 1).clamp_min(1e-9);
    return (e1 * e2).sum(1) / cross;  // [F]
  };
  const auto cot_a = 0.5 * cot_at(va, vb, vc);  // weight for opposite edge (b,c)
  const auto cot_b = 0.5 * cot_at(vb, vc, va);  // edge (c,a)
  const auto cot_c = 0.5 * cot_at(vc, va, vb);  // edge (a,b)
  // Accumulate off-diagonal -w and diagonal +w for each cotangent contribution (COO, coalesced).
  std::vector<Tensor> rows, cols, vals;
  auto add_edge = [&](const Tensor& i, const Tensor& j, const Tensor& w) {
    rows.insert(rows.end(), {i, j, i, j});
    cols.insert(cols.end(), {j, i, i, j});
    vals.insert(vals.end(), {-w, -w, w, w});
  };
  add_edge(b, c, cot_a);
  add_edge(c, a, cot_b);
  add_edge(a, b, cot_c);
  const auto idx = torch::stack({torch::cat(rows), torch::cat(cols)}, 0);  // [2, 12F]
  const auto val = torch::cat(vals);                                        // [12F]
  return torch::sparse_coo_tensor(idx, val, {V, V}).coalesce();
}

GeomResult solve_geometry(const Tensor& base_in, const Tensor& idb_in, const Tensor& exb_in,
                          const Tensor& faces_in, const Tensor& lap_in, const Tensor& assoc_in,
                          const Tensor& bary_in, const Tensor& lm_in, const Tensor& conf_in,
                          const Tensor& wprior_in, const GeomConfig& cfg) {
  const auto fopt = at::TensorOptions().dtype(at::kFloat);
  const auto base = base_in.to(at::kCPU, at::kFloat).contiguous();    // [V,3]
  const auto idb = idb_in.to(at::kCPU, at::kFloat).contiguous();      // [V,3,nid]
  const auto exb = exb_in.to(at::kCPU, at::kFloat).contiguous();      // [V,3,nex]
  const auto faces = faces_in.to(at::kCPU, at::kLong).contiguous();   // [F,3]
  const auto lap = lap_in.is_sparse() ? lap_in.to(at::kCPU, at::kFloat).coalesce()
                                      : lap_in.to(at::kCPU, at::kFloat).contiguous();  // [V,V]
  const auto assoc = assoc_in.to(at::kCPU, at::kLong).contiguous();   // [K]
  const auto bary = bary_in.to(at::kCPU, at::kFloat).contiguous();    // [K,3]
  const auto lm = lm_in.to(at::kCPU, at::kFloat).contiguous();        // [N,K,2]
  const auto conf = conf_in.to(at::kCPU, at::kFloat).contiguous();    // [N,K]
  const int64_t V = base.size(0), nid = idb.size(2), nex = exb.size(2);
  const int64_t N = lm.size(0), K = assoc.size(0);

  // corner vertex indices per dense point, and the per-point bases (barycentric gather).
  const auto corner = faces.index_select(0, assoc);                  // [K,3] vertex ids
  const auto cflat = corner.reshape({-1});                            // [K*3]
  auto bary_gather = [&](const Tensor& X) {  // X[V,...] -> [K,...] barycentric blend over corners
    const auto g = X.index_select(0, cflat);                         // [K*3, ...]
    if (X.dim() == 2)
      return torch::einsum("kc,kcd->kd", {bary, g.reshape({K, 3, X.size(1)})});
    return torch::einsum("kc,kcde->kde", {bary, g.reshape({K, 3, X.size(1), X.size(2)})});
  };
  const auto base_k = bary_gather(base);     // [K,3]
  const auto id_k = bary_gather(idb);        // [K,3,nid]
  const auto ex_k = bary_gather(exb);        // [K,3,nex]

  // Δv gather (vertex displacement -> per-point) and scatter (adjoint, per-point -> vertex).
  auto gather_dv = [&](const Tensor& dv) {   // dv[V,3] -> [K,3]
    return torch::einsum("kc,kcd->kd", {bary, dv.index_select(0, cflat).reshape({K, 3, 3})});
  };
  auto scatter_dv = [&](const Tensor& g) {   // g[K,3] -> [V,3]
    const auto contrib = (bary.unsqueeze(2) * g.unsqueeze(1)).reshape({K * 3, 3});  // [K*3,3]
    return torch::zeros({V, 3}, fopt).index_add_(0, cflat, contrib);
  };

  auto beta = torch::zeros({nid}, fopt);
  auto psi = torch::zeros({N, nex}, fopt);
  auto dv = torch::zeros({V, 3}, fopt);
  std::vector<Tensor> M(static_cast<size_t>(N)), t(static_cast<size_t>(N));
  auto w = wprior_in.defined() ? wprior_in.to(at::kCPU, at::kFloat).clone() : torch::ones({N}, fopt);

  // Posed dense points for current (β,ψ_i,Δv).
  auto shape_pt = [&](const Tensor& b, const Tensor& p, const Tensor& d) {
    return base_k + torch::einsum("kcn,n->kc", {id_k, b}) + torch::einsum("kcn,n->kc", {ex_k, p}) +
           gather_dv(d);
  };
  auto fit_pose = [&](const Tensor& q, const Tensor& u, const Tensor& cw, Tensor& Mi, Tensor& ti) {
    const auto sw = cw.sqrt().unsqueeze(1);                         // [K,1] per-point weight
    const auto D = torch::cat({q, torch::ones({K, 1}, fopt)}, 1) * sw;  // [K,4]
    const auto Mt = ridge_solve(D, u * sw, 1e-6);                   // [4,2]
    Mi = Mt.slice(0, 0, 3).t().contiguous();
    ti = Mt.slice(0, 3, 4).squeeze(0).contiguous();
  };
  for (int64_t i = 0; i < N; ++i)
    fit_pose(base_k, lm[i], conf[i], M[static_cast<size_t>(i)], t[static_cast<size_t>(i)]);

  double resid = 0;
  for (int it = 0; it < cfg.iterations; ++it) {
    // ---- per-photo pose + expression ----
    for (int64_t i = 0; i < N; ++i) {
      const auto si = static_cast<size_t>(i);
      fit_pose(shape_pt(beta, psi[i], dv), lm[i], conf[i], M[si], t[si]);
      const auto sid = base_k + torch::einsum("kcn,n->kc", {id_k, beta}) + gather_dv(dv);  // [K,3]
      const auto pred0 = torch::matmul(sid, M[si].t()) + t[si].unsqueeze(0);                // [K,2]
      const auto sw = conf[i].sqrt().unsqueeze(1);
      const auto r = ((lm[i] - pred0) * sw).reshape({K * 2});
      const auto E = (torch::einsum("ab,kbn->kan", {M[si], ex_k}) * sw.unsqueeze(2)).reshape({K * 2, nex});
      psi[i] = ridge_solve(E, r, cfg.expr_ridge);
    }
    // ---- shared β ----
    std::vector<Tensor> As, bs;
    for (int64_t i = 0; i < N; ++i) {
      const auto si = static_cast<size_t>(i);
      const auto sex = base_k + torch::einsum("kcn,n->kc", {ex_k, psi[i]}) + gather_dv(dv);
      const auto pred0 = torch::matmul(sex, M[si].t()) + t[si].unsqueeze(0);
      const auto sw = (conf[i] * std::max(1e-4F, w[i].item<float>())).sqrt().unsqueeze(1);
      As.push_back((torch::einsum("ab,kbn->kan", {M[si], id_k}) * sw.unsqueeze(2)).reshape({K * 2, nid}));
      bs.push_back(((lm[i] - pred0) * sw).reshape({K * 2}));
    }
    beta = ridge_solve(torch::cat(As, 0), torch::cat(bs, 0), cfg.id_ridge);

    // ---- shared Δv: (Σᵢ Gᵀ Mᵀ W M G + λ_lap LᵀL + λ_mag I) Δv = Σᵢ Gᵀ Mᵀ W r0  via matrix-free CG ----
    std::vector<Tensor> mm(static_cast<size_t>(N)), wt(static_cast<size_t>(N));
    auto rhs = torch::zeros({V, 3}, fopt);
    for (int64_t i = 0; i < N; ++i) {
      const auto si = static_cast<size_t>(i);
      mm[si] = torch::matmul(M[si].t(), M[si]);                     // [3,3]
      wt[si] = (conf[i] * std::max(1e-4F, w[i].item<float>())).unsqueeze(1);  // [K,1]
      const auto sfix = base_k + torch::einsum("kcn,n->kc", {id_k, beta}) +
                        torch::einsum("kcn,n->kc", {ex_k, psi[i]});  // q at Δv=0
      const auto r0 = lm[i] - (torch::matmul(sfix, M[si].t()) + t[si].unsqueeze(0));  // [K,2]
      rhs += scatter_dv(torch::matmul(r0, M[si]) * wt[si]);        // Mᵀr0 = r0·M (M is [2,3])
    }
    // L applied as a matvec (sparse-safe — never materialize LᵀL, which is dense [V,V]). L is
    // symmetric (cotangent/graph), so LᵀL·p = L·(L·p).
    auto Lmul = [&](const Tensor& x) {
      return lap.is_sparse() ? torch::mm(lap, x) : torch::matmul(lap, x);
    };
    auto apply = [&](const Tensor& p) {
      const auto gp = gather_dv(p);                                // [K,3]
      auto acc = torch::zeros({V, 3}, fopt);
      for (int64_t i = 0; i < N; ++i) {
        const auto si = static_cast<size_t>(i);
        acc += scatter_dv(torch::einsum("ab,kb->ka", {mm[si], gp}) * wt[si]);
      }
      return acc + cfg.lap_weight * Lmul(Lmul(p)) + cfg.mag_weight * p;
    };
    {  // conjugate gradient
      auto x = dv.clone();
      auto r = rhs - apply(x);
      auto p = r.clone();
      double rs = (r * r).sum().item<double>();
      for (int k = 0; k < cfg.cg_iters && rs > 1e-10; ++k) {
        const auto Ap = apply(p);
        const double alpha = rs / std::max((p * Ap).sum().item<double>(), 1e-20);
        x = x + alpha * p;
        r = r - alpha * Ap;
        const double rsn = (r * r).sum().item<double>();
        p = r + (rsn / rs) * p;
        rs = rsn;
      }
      dv = x;
    }

    // ---- robust per-photo reweight (C2) ----
    auto rms = torch::zeros({N}, fopt);
    for (int64_t i = 0; i < N; ++i) {
      const auto si = static_cast<size_t>(i);
      const auto pred = torch::matmul(shape_pt(beta, psi[i], dv), M[si].t()) + t[si].unsqueeze(0);
      rms[i] = ((lm[i] - pred).pow(2).sum(1) * conf[i]).sqrt().mean();
    }
    resid = rms.mean().item<double>();
    if (cfg.robust) {
      const auto c = (cfg.robust_k * rms.median()).clamp_min(1e-6);
      w = (wprior_in.defined() ? wprior_in.to(at::kCPU, at::kFloat) : torch::ones({N}, fopt)) *
          torch::exp(-0.5 * (rms / c).pow(2));
    }
  }

  // ---- observability field o(v) (M5): angular diversity of confident observers ----
  auto Tten = torch::zeros({V, 3, 3}, fopt);
  auto cover = torch::zeros({V}, fopt);
  for (int64_t i = 0; i < N; ++i) {
    const auto si = static_cast<size_t>(i);
    const auto r0 = M[si][0], r1 = M[si][1];                        // image-plane dirs in 3D
    auto axis = torch::cross(r0, r1);                              // optical axis
    axis = axis / axis.norm().clamp_min(1e-9);
    // per-vertex coverage in this photo = scattered confidence × trust.
    const auto cv = scatter_dv((conf[i] * std::max(0.0F, w[i].item<float>())).unsqueeze(1).expand({K, 3}))
                        .select(1, 0);                            // [V]
    Tten += cv.reshape({V, 1, 1}) * torch::matmul(axis.unsqueeze(1), axis.unsqueeze(0));
    cover += cv;
  }
  const auto ev = torch::linalg_eigvalsh(Tten + 1e-9 * torch::eye(3, fopt).unsqueeze(0));  // [V,3] asc
  const auto l1 = ev.select(1, 2).clamp_min(1e-9), l2 = ev.select(1, 1);
  const auto diversity = (l2 / l1).clamp(0.0, 1.0);                // ~1 if seen from ≥2 directions
  const auto o = torch::sigmoid(8.0 * (diversity * torch::tanh(cover) - 0.15));  // [V]
  dv = dv * (o > cfg.o_solve).to(fopt).unsqueeze(1);              // HARD gate

  GeomResult R;
  R.beta = beta;
  R.delta_v = dv;
  R.obs = o;
  R.expr = psi;
  R.weight = w;
  R.residual = resid;
  return R;
}

}  // namespace ncg::geom

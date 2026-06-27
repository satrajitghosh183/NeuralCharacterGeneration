#include <ncg/select/subject_gate.hpp>

#include <ncg/core/error.hpp>

#include <torch/torch.h>

namespace ncg::select {

namespace {
// L2-normalize rows of [P,D] (defensive — ArcFace outputs are already unit, but callers may not).
Tensor l2norm_rows(const Tensor& x) {
  return x / x.norm(2, /*dim=*/1, /*keepdim=*/true).clamp_min(1e-12);
}
}  // namespace

SubjectGate subject_gate(const Tensor& embeddings_in, const SubjectGateConfig& cfg) {
  NCG_CHECK(embeddings_in.dim() == 2, "subject_gate: embeddings must be [P,D]");
  const auto E = l2norm_rows(embeddings_in.to(at::kCPU).to(at::kFloat).contiguous());  // [P,D]
  const int64_t P = E.size(0);

  SubjectGate out;
  if (P == 0) {  // empty album
    out.is_subject = torch::zeros({0}, at::kBool);
    out.w_prior = torch::zeros({0}, at::kFloat);
    out.centroid = torch::zeros({E.size(1)}, at::kFloat);
    return out;
  }

  // Seed at the densest embedding: the face with the most same-identity neighbors. The subject is
  // the dominant mode because the real person recurs across the album while incidental others don't.
  const auto S = torch::matmul(E, E.t());                      // [P,P] cosine similarity
  const auto neigh = (S >= cfg.same_id_cos).to(at::kFloat);    // [P,P]
  const int64_t seed = std::get<1>(neigh.sum(1).max(0)).item<int64_t>();

  // Robust mode-seeking: trimmed-mean centroid over the seed neighborhood, refined to convergence.
  auto members = (S[seed] >= cfg.same_id_cos);                 // [P] bool, seed's neighborhood
  auto mu = l2norm_rows(E.index({members}).mean(0, /*keepdim=*/true)).squeeze(0);  // [D]
  for (int it = 0; it < cfg.iterations; ++it) {
    const auto cos = torch::matmul(E, mu);                     // [P]
    members = cos >= cfg.keep_cos;
    if (members.sum().item<int64_t>() == 0) { members = (S[seed] >= cfg.same_id_cos); break; }
    mu = l2norm_rows(E.index({members}).mean(0, true)).squeeze(0);
  }

  const auto cos = torch::matmul(E, mu);                       // [P] cosine to subject centroid
  const auto is_sub = cos >= cfg.keep_cos;                     // [P] bool
  const auto dist = (1.0 - cos);                               // cosine distance

  // Robust trust: Welsch on the deviation of each face's distance from the TYPICAL subject distance
  // (median), one-sided — only a face FARTHER than typical is suspect; a tightly-clustered subject
  // face sits at the baseline distance ⇒ deviation≈0 ⇒ trust≈1. (Penalizing the raw distance would
  // wrongly zero out every subject face, since on the unit sphere the baseline distance dwarfs the
  // cluster spread.)  Non-subject faces get exactly 0.
  double mad = 0, med_d = 0;
  const auto sub_d = dist.index({is_sub});
  if (sub_d.numel() > 0) {
    med_d = sub_d.median().item<double>();
    mad = 1.4826 * (sub_d - med_d).abs().median().item<double>();
  }
  const double c = cfg.robust_k * std::max(mad, 1e-6);
  const auto dev = (dist - med_d).clamp_min(0.0);  // farther-than-typical ⇒ down-weight
  const auto w = torch::exp(-0.5 * (dev / c).pow(2)) * is_sub.to(at::kFloat);

  out.is_subject = is_sub;
  out.w_prior = w;
  out.centroid = mu;
  out.mad = mad;
  out.n_subject = is_sub.sum().item<int64_t>();
  return out;
}

}  // namespace ncg::select

#include <catch2/catch_test_macros.hpp>

#include <ncg/select/subject_gate.hpp>

#include <torch/torch.h>

// Phase A / C2 proof on synthetic ground truth: an album of ArcFace-style embeddings where the
// SUBJECT recurs across many photos and several OTHER people appear incidentally. The gate must
// recover the subject as the dominant identity, reject every other-person face (precision/recall),
// and assign a high trust prior to the subject and zero to contamination — all WITHOUT any real
// ArcFace weights. This is the contamination-robustness result (docs/method.md §M3).

namespace {
constexpr int64_t D = 512;  // ArcFace embedding dim

// One face embedding near identity centroid `mu`, with realistic intra-identity jitter.
torch::Tensor face(const torch::Tensor& mu, float jitter) {
  auto e = mu + jitter * torch::randn({D});
  return e / e.norm().clamp_min(1e-9);
}
torch::Tensor identity() { return face(torch::randn({D}), 0.0F); }  // a person's centroid (unit)
}  // namespace

TEST_CASE("subject_gate: dominant identity recovered, contamination rejected (C2)", "[select][gate]") {
  torch::manual_seed(0);
  const float jit = 0.03F;  // intra-identity spread (keeps subject faces cos>keep_cos)

  const auto mu_subject = identity();
  std::vector<torch::Tensor> rows;
  std::vector<bool> is_subject_gt;

  const int N_SUB = 40;  // subject appears in 40 photos (the dominant mode)
  for (int i = 0; i < N_SUB; ++i) { rows.push_back(face(mu_subject, jit)); is_subject_gt.push_back(true); }

  // Inject 4 other people, 5 incidental faces each = 20 contamination faces.
  const int N_OTHER_PEOPLE = 4, FACES_EACH = 5;
  for (int p = 0; p < N_OTHER_PEOPLE; ++p) {
    const auto mu_other = identity();
    for (int j = 0; j < FACES_EACH; ++j) { rows.push_back(face(mu_other, jit)); is_subject_gt.push_back(false); }
  }

  const auto embeddings = torch::stack(rows, 0);  // [P,D]
  const auto g = ncg::select::subject_gate(embeddings);

  // Precision / recall of subject classification.
  int tp = 0, fp = 0, fn = 0, tn = 0;
  const auto is_sub = g.is_subject.to(at::kCPU);
  for (size_t i = 0; i < is_subject_gt.size(); ++i) {
    const bool pred = is_sub[static_cast<int64_t>(i)].item<bool>();
    if (is_subject_gt[i] && pred) ++tp;
    else if (!is_subject_gt[i] && pred) ++fp;
    else if (is_subject_gt[i] && !pred) ++fn;
    else ++tn;
  }
  const double precision = tp / static_cast<double>(tp + fp + 1e-9);
  const double recall = tp / static_cast<double>(tp + fn + 1e-9);
  INFO("gate P=" << precision << " R=" << recall << "  subject kept=" << g.n_subject
                 << "  (tp=" << tp << " fp=" << fp << " fn=" << fn << " tn=" << tn << ")");

  REQUIRE(g.n_subject == N_SUB);  // exactly the subject faces, no more no less
  REQUIRE(precision == 1.0);      // no other-person face slips through
  REQUIRE(recall == 1.0);         // no subject face dropped

  // Trust prior: high for subject, exactly zero for contamination.
  const auto w = g.w_prior.to(at::kCPU);
  double w_sub = 0, w_bad = 0;
  for (size_t i = 0; i < is_subject_gt.size(); ++i)
    (is_subject_gt[i] ? w_sub : w_bad) += w[static_cast<int64_t>(i)].item<double>();
  REQUIRE(w_sub / N_SUB > 0.5);                                  // subject genuinely trusted
  REQUIRE(w_bad == 0.0);                                         // contamination fully gated out
}

TEST_CASE("subject_gate: single-photo album degenerates gracefully", "[select][gate]") {
  torch::manual_seed(1);
  const auto e = (torch::randn({1, D}));
  const auto g = ncg::select::subject_gate(e);
  REQUIRE(g.n_subject == 1);
  REQUIRE(g.is_subject[0].item<bool>());
}

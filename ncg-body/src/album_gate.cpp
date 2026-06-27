#include <ncg/body/album_gate.hpp>

#include <ncg/core/error.hpp>
#include <ncg/core/logging.hpp>
#include <ncg/io/image.hpp>
#include <ncg/select/subject_gate.hpp>

#include <torch/torch.h>

#include <vector>

namespace ncg::body {
namespace {
// One detected face with bookkeeping back to its source photo (for the album-level gate).
struct Det {
  size_t photo;
  FaceBox box;
  Tensor emb;  // [512]
};
}  // namespace

std::vector<PhotoBundle> gate_album(const std::vector<std::string>& paths,
                                    const AlbumGateModels& models, const AlbumGateConfig& cfg) {
  NCG_CHECK(models.detector && models.mesh && models.arcface,
            "gate_album: AlbumGateModels must have detector, mesh and arcface set");
  const size_t P = paths.size();
  std::vector<PhotoBundle> out(P);
  for (size_t i = 0; i < P; ++i) out[i].path = paths[i];

  // ---- Pass 1: detect ALL faces in every photo, embed each (identity) ----
  std::vector<Det> dets;
  for (size_t i = 0; i < P; ++i) {
    Tensor img;
    try {
      img = ncg::io::load_image(paths[i], 3);
    } catch (const std::exception& e) {
      NCG_LOG_WARN("gate_album: cannot read '{}' ({}) — skipping", paths[i], e.what());
      continue;
    }
    std::vector<FaceBox> faces;
    try {
      faces = models.detector->detect(img, cfg.min_det_score);
    } catch (const std::exception& e) {
      NCG_LOG_WARN("gate_album: detector failed on '{}' ({})", paths[i], e.what());
      continue;
    }
    for (auto& b : faces) {
      Tensor emb;
      try {
        emb = models.arcface->embed(img, b);
      } catch (const std::exception& e) {
        NCG_LOG_WARN("gate_album: arcface failed on a face in '{}' ({})", paths[i], e.what());
        continue;
      }
      dets.push_back(Det{i, std::move(b), std::move(emb)});
    }
  }

  if (dets.empty()) {
    NCG_LOG_WARN("gate_album: no faces detected in any of {} photos", P);
    return out;
  }

  // ---- Album-level identity gate: dominant cluster = subject, others = contamination ----
  std::vector<Tensor> embs;
  embs.reserve(dets.size());
  for (const auto& d : dets) embs.push_back(d.emb);
  const auto gate = ncg::select::subject_gate(torch::stack(embs, 0), cfg.subject);
  const auto is_sub = gate.is_subject.to(at::kCPU);
  const auto w_prior = gate.w_prior.to(at::kCPU);
  const auto cos = torch::matmul(torch::stack(embs, 0), gate.centroid.unsqueeze(1)).squeeze(1).to(at::kCPU);

  // ---- Pass 2: per photo, pick the most-trusted subject face → dense landmarks; log the rest ----
  // Best subject detection index per photo (highest trust), and rejected faces.
  std::vector<int64_t> best(P, -1);
  for (size_t m = 0; m < dets.size(); ++m) {
    if (!is_sub[static_cast<int64_t>(m)].item<bool>()) continue;
    const size_t i = dets[m].photo;
    if (best[i] < 0 || w_prior[static_cast<int64_t>(m)].item<float>() >
                           w_prior[best[i]].item<float>())
      best[i] = static_cast<int64_t>(m);
  }

  int64_t n_usable = 0, n_rejected = 0;
  for (size_t i = 0; i < P; ++i) {
    // Record every non-chosen face in this photo as rejected (audit trail for the C2 claim).
    for (size_t m = 0; m < dets.size(); ++m) {
      if (dets[m].photo != i) continue;
      if (static_cast<int64_t>(m) == best[i]) continue;
      RejectedFace rf;
      rf.path = paths[i];
      rf.bbox = dets[m].box.bbox;
      rf.det_score = dets[m].box.score;
      rf.embed_dist = 1.0F - cos[static_cast<int64_t>(m)].item<float>();
      rf.reason = is_sub[static_cast<int64_t>(m)].item<bool>() ? "secondary-subject-face" : "other-person";
      out[i].rejected.push_back(std::move(rf));
      ++n_rejected;
    }
    if (best[i] < 0) { out[i].usable = false; continue; }  // no subject face in this photo

    Tensor img;
    try {
      img = ncg::io::load_image(paths[i], 3);
    } catch (const std::exception& e) {
      NCG_LOG_WARN("gate_album: re-read '{}' failed ({}) — marking unusable", paths[i], e.what());
      out[i].usable = false;
      continue;
    }
    const auto& box = dets[best[i]].box;
    DenseLandmarks dl;
    try {
      dl = models.mesh->mesh(img, box);
    } catch (const std::exception& e) {
      NCG_LOG_WARN("gate_album: facemesh failed on '{}' ({}) — unusable", paths[i], e.what());
      out[i].usable = false;
      continue;
    }
    // Zero-weight low-confidence dense points (consumed by the Phase B solve).
    out[i].dense_uv = dl.uv;
    out[i].dense_conf = dl.conf * (dl.conf >= cfg.min_landmark_conf).to(at::kFloat);
    out[i].subject_bbox = box.bbox;
    out[i].w_prior = w_prior[best[i]].item<float>();
    out[i].usable = true;
    ++n_usable;
  }

  NCG_LOG_INFO("gate_album: {} photos → {} usable subject faces, {} faces rejected "
               "({} subject embeddings, robust scale {:.4f})",
               P, n_usable, n_rejected, gate.n_subject, gate.mad);
  return out;
}

}  // namespace ncg::body

#pragma once

#include <ncg/body/face_models.hpp>
#include <ncg/core/tensor.hpp>

#include <string>
#include <vector>

namespace ncg::body {

// ============================================================================================
// PHASE A INTERFACE CONTRACT (frozen). Turns a dirty, contaminated album into per-photo bundles
// carrying ONLY the subject's dense face landmarks + a robust trust prior (docs/method.md §M2/§M3).
// Pipeline per photo: FaceDetector → (all faces) → ArcFace embed each → album-level
// select::subject_gate picks the dominant identity → FaceMeshNet on the subject face → dense_uv.
// Photos with other people still contribute their subject face; every rejected face is logged.
// Downstream (Phase B solve_geometry) consumes std::vector<PhotoBundle> and nothing else.
// ============================================================================================

/// A face that was detected but NOT used, with the reason (audit trail for the C2 claim).
struct RejectedFace {
  std::string path;       // source photo
  Tensor bbox;            // [4] pixel bbox
  float det_score = 0;    // detector confidence
  float embed_dist = 0;   // cosine distance to the subject centroid (large ⇒ different person)
  std::string reason;     // "other-person" | "no-subject-face" | "low-detection" | "no-landmarks"
};

/// Per-photo result. `usable=false` ⇒ no subject face found (photo skipped by the solve).
struct PhotoBundle {
  std::string path;
  Tensor dense_uv;        // [K,2] subject-face dense landmarks (canonical FaceMesh order), image px
  Tensor dense_conf;      // [K]   per-point confidence γ ∈ [0,1]
  Tensor subject_crop;    // [3,H,W] the subject face crop (for downstream appearance/recheck)
  Tensor subject_bbox;    // [4] pixel bbox of the chosen subject face
  float w_prior = 0;      // Welsch trust from embedding distance to subject centroid (§M3)
  bool usable = false;
  std::vector<RejectedFace> rejected;  // other-person / junk faces in THIS photo
};

/// Ported nets the gate runs (loaded once, shared across the album).
struct AlbumGateModels {
  const FaceDetector* detector = nullptr;
  const FaceMeshNet* mesh = nullptr;
  const ArcFace* arcface = nullptr;
};

struct AlbumGateConfig {
  float min_det_score = 0.5F;   // discard detections below this
  float min_landmark_conf = 0.3F;  // dense points below γ are zero-weighted downstream
  ncg::select::SubjectGateConfig subject;  // robust identity-clustering params (§M3)
};

/// PHASE A entry point. Runs detect→embed→subject-gate→landmark over `frame_paths`, returning one
/// PhotoBundle per photo (subject landmarks + trust + rejected faces). Album-level identity
/// clustering is delegated to ncg::select::subject_gate. Logs a one-line summary of kept/rejected
/// faces and the contamination count (the C2 result). Does NOT throw on a per-photo failure — it
/// marks that bundle unusable and continues (dirty-input robustness is the point).
std::vector<PhotoBundle> gate_album(const std::vector<std::string>& frame_paths,
                                    const AlbumGateModels& models, const AlbumGateConfig& cfg = {});

}  // namespace ncg::body

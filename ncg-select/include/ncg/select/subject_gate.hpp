#pragma once

#include <ncg/core/tensor.hpp>

namespace ncg::select {

// ============================================================================================
// Album-level subject gate — the C2 firewall (docs/method.md §M3). The casual album contains
// OTHER PEOPLE; before any shared-identity solve we must keep only the subject's face in each
// photo and down-weight marginal observations. Given one ArcFace embedding per detected face over
// the WHOLE album, find the dominant identity cluster (the subject, who appears in most photos),
// flag every other face as contamination, and emit a robust per-face trust prior.
//
// This is a PURE function of the embeddings (no net dependency) so it lives in ncg-select and is
// proven on synthetic ground truth (inject N wrong-person embeddings → assert all gated) before any
// real ArcFace weights exist. ncg-body::gate_album wraps it with the ported detector/embedder.
// ============================================================================================

struct SubjectGateConfig {
  // Two faces are "the same person" when cosine similarity exceeds this (ArcFace convention).
  float same_id_cos = 0.5F;
  // Welsch trust scale = robust_k · (1.4826·MAD of subject distances). Larger ⇒ gentler down-weight.
  float robust_k = 3.0F;
  // Mode-seeking refinement iterations for the robust subject centroid.
  int iterations = 8;
  // A face is kept as the subject iff cosine(e, μ_subject) ≥ keep_cos (after refinement).
  float keep_cos = 0.45F;
};

/// Result of gating P detected faces (across the whole album) by identity.
struct SubjectGate {
  Tensor is_subject;  // [P] bool — face p belongs to the dominant (subject) cluster
  Tensor w_prior;     // [P] float in [0,1] — Welsch trust from embedding distance (0 for non-subject)
  Tensor centroid;    // [D] float — L2-normalized subject identity centroid μ_D
  double mad = 0;     // robust scale (1.4826·MAD) of subject cosine distances
  int64_t n_subject = 0;
};

/// Robust dominant-identity gate. `embeddings` [P,D] are L2-normalized ArcFace vectors for every
/// detected face in the album (any photo, any person). Returns which faces are the subject, a
/// Welsch trust prior per face, and the subject centroid. Algorithm (robust mode-seeking on the
/// unit hypersphere, no HDBSCAN dependency): seed at the densest embedding (most same-id neighbors),
/// iterate a trimmed-mean centroid over its neighborhood, then keep faces within `keep_cos`. The
/// subject is identifiable as the *dominant* mode because the real person recurs across the album
/// while incidental others do not. `owner` maps each face row → its source photo (for logging).
SubjectGate subject_gate(const Tensor& embeddings, const SubjectGateConfig& cfg = {});

}  // namespace ncg::select

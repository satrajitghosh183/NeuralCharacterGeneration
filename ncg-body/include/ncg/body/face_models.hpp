#pragma once

#include <ncg/core/tensor.hpp>

#include <memory>
#include <string>
#include <vector>

namespace ncg::body {

// ============================================================================================
// Ported face front-end nets for Phase A (docs/method.md §M2/§M3). All three are PORTED (load
// published weights, prove forward-pass parity per docs/parity.md) — NOT trained. They are thin
// TorchScript loaders (same pattern as ncg::body::Nlf): load once, run a forward, no Python at
// runtime. Parity golden tests SKIP when the model asset is absent (assets are git-LFS / fetched).
//   • FaceDetector  — find ALL faces in a photo (album may contain other people).
//   • FaceMeshNet   — K≈468 dense 2D landmarks + per-point confidence for one face crop.
//   • ArcFace       — a 512-d L2-normalized identity embedding for one face crop.
// ============================================================================================

/// One detected face: pixel bbox (x0,y0,x1,y1), detector score, and 6 facial keypoints
/// (right eye, left eye, nose, mouth, R-ear, L-ear) in image pixels — used to ALIGN the face
/// (similarity transform to a canonical template) before the identity embedder. `kpts` may be
/// undefined if the detector did not provide keypoints (then embedding falls back to a bbox crop).
struct FaceBox {
  Tensor bbox;        // [4] float, image pixels (x0,y0,x1,y1)
  float score = 0;    // detector confidence
  Tensor kpts;        // [6,2] float image pixels, or undefined
};

/// K dense facial landmarks for one face, in the canonical FaceMesh point order (so the fixed
/// FaceMesh→SMPL-X embedding from Phase B applies). `uv` are image-space pixels.
struct DenseLandmarks {
  Tensor uv;    // [K,2] float, image pixels
  Tensor conf;  // [K]   float in [0,1], per-point confidence γ
};

/// Multi-face detector (e.g. BlazeFace / RetinaFace), TorchScript. Returns every face above
/// `min_score`, sorted by score descending. Detects ALL people, not just the largest/centered.
class FaceDetector {
public:
  static FaceDetector load(const std::string& torchscript_path, at::Device device);
  std::vector<FaceBox> detect(const Tensor& image_chw, float min_score = 0.5F) const;
  FaceDetector(FaceDetector&&) noexcept;
  ~FaceDetector();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  explicit FaceDetector(std::unique_ptr<Impl> impl);
};

/// Dense landmark mesh (e.g. MediaPipe FaceMesh, 468 pts), TorchScript. Given a face crop region,
/// returns landmarks in image-space pixels + confidence, in canonical point order.
class FaceMeshNet {
public:
  static FaceMeshNet load(const std::string& torchscript_path, at::Device device);
  DenseLandmarks mesh(const Tensor& image_chw, const FaceBox& box) const;
  int64_t num_points() const;
  FaceMeshNet(FaceMeshNet&&) noexcept;
  ~FaceMeshNet();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  explicit FaceMeshNet(std::unique_ptr<Impl> impl);
};

/// ArcFace identity embedder, TorchScript. Returns a 512-d L2-normalized embedding for a face crop.
class ArcFace {
public:
  static ArcFace load(const std::string& torchscript_path, at::Device device);
  Tensor embed(const Tensor& image_chw, const FaceBox& box) const;  // [512], L2-normalized
  ArcFace(ArcFace&&) noexcept;
  ~ArcFace();

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  explicit ArcFace(std::unique_ptr<Impl> impl);
};

}  // namespace ncg::body

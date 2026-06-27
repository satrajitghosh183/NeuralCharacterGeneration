#include <ncg/body/face_models.hpp>

#include <ncg/core/error.hpp>
#include <ncg/core/logging.hpp>

#include <torch/script.h>
#include <torch/torch.h>

#include <algorithm>
#include <utility>

// Ported face front-end nets (Phase A). Each is a TorchScript module loaded with torch::jit::load
// (same proven pattern as ncg::body::Nlf — no Python at runtime). The C++ here is model-AGNOSTIC:
// the offline export tool (tools/export_face_models.py) wraps whatever published model into a module
// matching a FIXED I/O contract, so this code never depends on a specific architecture:
//   • detector.forward(img[3,H,W] f32 in [0,1])      -> boxes [N,5] = (x0,y0,x1,y1,score), px
//   • facemesh.forward(crop[3,192,192] f32 in [0,1]) -> (uv[K,2] in [0,1] crop, conf[K])
//   • arcface.forward(crop[3,112,112] f32 in [0,1])  -> emb[512] (L2-normalized)

namespace ncg::body {
namespace {

namespace F = torch::nn::functional;

// Crop the pixel box from a CHW image and resize to out×out (bilinear). Box clamped to bounds.
Tensor crop_resize(const Tensor& img, const Tensor& box, int64_t out) {
  const int64_t H = img.size(1), W = img.size(2);
  const auto b = box.to(at::kCPU).to(at::kFloat);
  const int64_t x0 = std::clamp<int64_t>(static_cast<int64_t>(b[0].item<float>()), 0, W - 1);
  const int64_t y0 = std::clamp<int64_t>(static_cast<int64_t>(b[1].item<float>()), 0, H - 1);
  const int64_t x1 = std::clamp<int64_t>(static_cast<int64_t>(b[2].item<float>()), x0 + 1, W);
  const int64_t y1 = std::clamp<int64_t>(static_cast<int64_t>(b[3].item<float>()), y0 + 1, H);
  const auto patch = img.slice(1, y0, y1).slice(2, x0, x1);  // [3,bh,bw]
  return F::interpolate(patch.unsqueeze(0),
                        F::InterpolateFuncOptions()
                            .size(std::vector<int64_t>{out, out})
                            .mode(torch::kBilinear)
                            .align_corners(false))
      .squeeze(0);
}

// Umeyama similarity (scale·R + t) mapping src 2D points to dst (least-squares). src,dst [N,2].
// Returns the 2x3 affine [scale·R | t] s.t. dst ≈ A·[src;1].
Tensor umeyama(const Tensor& src, const Tensor& dst) {
  const int64_t n = src.size(0);
  const auto mu_s = src.mean(0), mu_d = dst.mean(0);
  const auto sc = src - mu_s, dc = dst - mu_d;
  const double var_s = sc.pow(2).sum().item<double>() / static_cast<double>(n);
  const auto Sigma = torch::matmul(dc.t(), sc) / static_cast<double>(n);  // [2,2]
  const auto usv = torch::linalg_svd(Sigma, /*full_matrices=*/false);
  const auto U = std::get<0>(usv), S = std::get<1>(usv), Vh = std::get<2>(usv);
  const double d1 =
      (torch::linalg_det(U).item<double>() * torch::linalg_det(Vh).item<double>() < 0) ? -1.0 : 1.0;
  const auto D = torch::tensor({1.0, d1}, src.options());
  const auto R = torch::matmul(U * D.unsqueeze(0), Vh);                   // U·diag(D)·Vh
  const double scale = (S * D).sum().item<double>() / std::max(var_s, 1e-9);
  const auto sR = scale * R;                                             // [2,2]
  const auto t = mu_d - torch::matmul(sR, mu_s.unsqueeze(1)).squeeze(1);  // [2]
  return torch::cat({sR, t.unsqueeze(1)}, 1);                            // [2,3]
}

// Align a face to a canonical 112² template using its keypoints (eyes, nose, mouth) — the standard
// preprocessing the identity embedder needs (raw bbox crops give unreliable embeddings). Warps the
// source image into canonical space via the inverse similarity + grid_sample.
Tensor align_crop(const Tensor& img, const Tensor& kpts, int64_t out = 112) {
  const auto opt = img.options();
  // Canonical positions for [right-eye, left-eye, nose, mouth-center] at out=112.
  const auto dst = torch::tensor({{38.0, 52.0}, {74.0, 52.0}, {56.0, 72.0}, {56.0, 92.0}}, opt);
  const auto src = kpts.slice(0, 0, 4).to(opt);                          // first 4 keypoints
  const auto A = umeyama(src, dst);                                      // [2,3] src->dst
  const auto sR = A.slice(1, 0, 2), t = A.slice(1, 2, 3).squeeze(1);
  const auto sRinv = torch::linalg_inv(sR);                             // canon->src rotation
  const auto tinv = -torch::matmul(sRinv, t.unsqueeze(1)).squeeze(1);
  const auto rng = torch::arange(out, opt);
  const auto xs = rng.unsqueeze(0).expand({out, out});
  const auto ys = rng.unsqueeze(1).expand({out, out});
  const auto cg = torch::stack({xs, ys}, 2).reshape({-1, 2});           // [out²,2] canonical px (x,y)
  const auto srcpx = torch::matmul(cg, sRinv.t()) + tinv;               // [out²,2] source px
  const int64_t H = img.size(1), W = img.size(2);
  const auto gx = srcpx.select(1, 0) / (W - 1) * 2 - 1;
  const auto gy = srcpx.select(1, 1) / (H - 1) * 2 - 1;
  const auto grid = torch::stack({gx, gy}, 1).reshape({1, out, out, 2});
  return F::grid_sample(img.unsqueeze(0), grid,
                        F::GridSampleFuncOptions().mode(torch::kBilinear)
                            .padding_mode(torch::kZeros).align_corners(true))
      .squeeze(0);
}

}  // namespace

// ---------------------------------------------------------------------------------------------
struct FaceDetector::Impl {
  torch::jit::Module module;
  at::Device device{at::kCPU};
};
FaceDetector::FaceDetector(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
FaceDetector::FaceDetector(FaceDetector&&) noexcept = default;
FaceDetector::~FaceDetector() = default;

FaceDetector FaceDetector::load(const std::string& path, at::Device device) {
  auto impl = std::make_unique<Impl>();
  impl->device = device;
  try {
    impl->module = torch::jit::load(path, device);
    impl->module.eval();
  } catch (const std::exception& e) {
    NCG_THROW("FaceDetector::load failed for '{}': {}", path, e.what());
  }
  NCG_LOG_INFO("FaceDetector: loaded '{}'", path);
  return FaceDetector(std::move(impl));
}

std::vector<FaceBox> FaceDetector::detect(const Tensor& image_chw, float min_score) const {
  torch::NoGradGuard ng;
  const auto img = image_chw.to(impl_->device).to(at::kFloat);
  auto out = impl_->module.forward({img}).toTensor().to(at::kCPU).contiguous();  // [N, 5+12]
  std::vector<FaceBox> boxes;
  const bool has_kp = out.size(1) >= 17;
  for (int64_t i = 0; i < out.size(0); ++i) {
    const float score = out[i][4].item<float>();
    if (score < min_score) continue;
    FaceBox fb;
    fb.bbox = out[i].slice(0, 0, 4).clone();
    fb.score = score;
    if (has_kp) fb.kpts = out[i].slice(0, 5, 17).clone().reshape({6, 2});  // 6 keypoints (x,y) px
    boxes.push_back(std::move(fb));
  }
  std::sort(boxes.begin(), boxes.end(), [](const FaceBox& a, const FaceBox& b) { return a.score > b.score; });
  return boxes;
}

// ---------------------------------------------------------------------------------------------
struct FaceMeshNet::Impl {
  torch::jit::Module module;
  at::Device device{at::kCPU};
  int64_t k = 0;
};
FaceMeshNet::FaceMeshNet(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
FaceMeshNet::FaceMeshNet(FaceMeshNet&&) noexcept = default;
FaceMeshNet::~FaceMeshNet() = default;

FaceMeshNet FaceMeshNet::load(const std::string& path, at::Device device) {
  auto impl = std::make_unique<Impl>();
  impl->device = device;
  try {
    impl->module = torch::jit::load(path, device);
    impl->module.eval();
  } catch (const std::exception& e) {
    NCG_THROW("FaceMeshNet::load failed for '{}': {}", path, e.what());
  }
  NCG_LOG_INFO("FaceMeshNet: loaded '{}'", path);
  return FaceMeshNet(std::move(impl));
}

int64_t FaceMeshNet::num_points() const { return impl_->k; }

DenseLandmarks FaceMeshNet::mesh(const Tensor& image_chw, const FaceBox& box) const {
  torch::NoGradGuard ng;
  constexpr int64_t IN = 192;
  const auto crop = crop_resize(image_chw.to(impl_->device).to(at::kFloat), box.bbox, IN).unsqueeze(0);
  const auto out = impl_->module.forward({crop});
  const auto tup = out.toTuple();
  auto uv = tup->elements()[0].toTensor().to(at::kCPU).to(at::kFloat).reshape({-1, 2});  // [K,2] in [0,1]
  auto conf = tup->elements()[1].toTensor().to(at::kCPU).to(at::kFloat).reshape({-1});    // [K]
  impl_->k = uv.size(0);
  // Map crop-normalized [0,1] landmarks back to source-image pixel coordinates.
  const auto b = box.bbox.to(at::kCPU).to(at::kFloat);
  const float x0 = b[0].item<float>(), y0 = b[1].item<float>();
  const float w = b[2].item<float>() - x0, h = b[3].item<float>() - y0;
  const auto px = uv.select(1, 0) * w + x0;
  const auto py = uv.select(1, 1) * h + y0;
  DenseLandmarks dl;
  dl.uv = torch::stack({px, py}, 1).contiguous();
  dl.conf = conf.clamp(0.0F, 1.0F);
  return dl;
}

// ---------------------------------------------------------------------------------------------
struct ArcFace::Impl {
  torch::jit::Module module;
  at::Device device{at::kCPU};
};
ArcFace::ArcFace(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ArcFace::ArcFace(ArcFace&&) noexcept = default;
ArcFace::~ArcFace() = default;

ArcFace ArcFace::load(const std::string& path, at::Device device) {
  auto impl = std::make_unique<Impl>();
  impl->device = device;
  try {
    impl->module = torch::jit::load(path, device);
    impl->module.eval();
  } catch (const std::exception& e) {
    NCG_THROW("ArcFace::load failed for '{}': {}", path, e.what());
  }
  NCG_LOG_INFO("ArcFace: loaded '{}'", path);
  return ArcFace(std::move(impl));
}

Tensor ArcFace::embed(const Tensor& image_chw, const FaceBox& box) const {
  torch::NoGradGuard ng;
  constexpr int64_t IN = 112;
  const auto img = image_chw.to(impl_->device).to(at::kFloat);
  // Keypoint alignment (standard ArcFace preprocessing) when available; bbox crop otherwise.
  const auto crop = (box.kpts.defined() && box.kpts.numel() >= 8)
                        ? align_crop(img, box.kpts.to(img.device()), IN).unsqueeze(0)
                        : crop_resize(img, box.bbox, IN).unsqueeze(0);
  auto e = impl_->module.forward({crop}).toTensor().to(at::kCPU).to(at::kFloat).reshape({-1});  // [512]
  return e / e.norm().clamp_min(1e-9);  // defensive L2-normalize
}

}  // namespace ncg::body

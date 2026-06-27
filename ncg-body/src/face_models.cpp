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
  auto out = impl_->module.forward({img}).toTensor().to(at::kCPU).contiguous();  // [N,5]
  std::vector<FaceBox> boxes;
  for (int64_t i = 0; i < out.size(0); ++i) {
    const float score = out[i][4].item<float>();
    if (score < min_score) continue;
    boxes.push_back(FaceBox{out[i].slice(0, 0, 4).clone(), score});
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
  const auto crop = crop_resize(image_chw.to(impl_->device).to(at::kFloat), box.bbox, IN).unsqueeze(0);
  auto e = impl_->module.forward({crop}).toTensor().to(at::kCPU).to(at::kFloat).reshape({-1});  // [512]
  return e / e.norm().clamp_min(1e-9);  // defensive L2-normalize
}

}  // namespace ncg::body

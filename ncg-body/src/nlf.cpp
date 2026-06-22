#include <ncg/body/nlf.hpp>

#include <ncg/core/error.hpp>
#include <ncg/core/logging.hpp>

#include <torch/library.h>
#include <torch/script.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ncg::body {

namespace {

// NLF's multi-person detector graph calls a single torchvision custom op: torchvision::nms.
// Rather than dlopen torchvision's _C.so (a Python extension that drags in libtorch_python +
// libpython and crashes when loaded into a non-Python process), we provide our own
// implementation and register it under the torchvision:: namespace below. Non-max suppression
// over a few hundred detection boxes is trivial, so a CPU O(n^2) sweep is plenty; indices are
// returned on the input's device. Schema matches torchvision exactly:
//   torchvision::nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor
Tensor nms_cpu(const Tensor& dets_in, const Tensor& scores_in, double iou_threshold) {
  TORCH_CHECK(dets_in.dim() == 2 && dets_in.size(1) == 4, "nms: dets must be [N,4]");
  TORCH_CHECK(scores_in.dim() == 1 && scores_in.size(0) == dets_in.size(0),
              "nms: scores must be [N] matching dets");
  const auto dets = dets_in.to(at::kCPU, at::kFloat).contiguous();
  const auto scores = scores_in.to(at::kCPU, at::kFloat).contiguous();
  const int64_t n = dets.size(0);
  auto keep = torch::empty({n}, at::TensorOptions().dtype(at::kLong));
  if (n == 0) return keep.to(dets_in.device());

  const float* d = dets.data_ptr<float>();
  const float* s = scores.data_ptr<float>();
  std::vector<int64_t> order(static_cast<size_t>(n));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return s[a] > s[b]; });

  std::vector<char> suppressed(static_cast<size_t>(n), 0);
  int64_t* k = keep.data_ptr<int64_t>();
  int64_t num_keep = 0;
  for (int64_t ii = 0; ii < n; ++ii) {
    const int64_t i = order[static_cast<size_t>(ii)];
    if (suppressed[static_cast<size_t>(i)]) continue;
    k[num_keep++] = i;
    const float ix1 = d[i * 4 + 0], iy1 = d[i * 4 + 1], ix2 = d[i * 4 + 2], iy2 = d[i * 4 + 3];
    const float iarea = (ix2 - ix1) * (iy2 - iy1);
    for (int64_t jj = ii + 1; jj < n; ++jj) {
      const int64_t j = order[static_cast<size_t>(jj)];
      if (suppressed[static_cast<size_t>(j)]) continue;
      const float xx1 = std::max(ix1, d[j * 4 + 0]);
      const float yy1 = std::max(iy1, d[j * 4 + 1]);
      const float xx2 = std::min(ix2, d[j * 4 + 2]);
      const float yy2 = std::min(iy2, d[j * 4 + 3]);
      const float w = std::max(0.0F, xx2 - xx1);
      const float h = std::max(0.0F, yy2 - yy1);
      const float inter = w * h;
      const float jarea = (d[j * 4 + 2] - d[j * 4 + 0]) * (d[j * 4 + 3] - d[j * 4 + 1]);
      const float iou = inter / (iarea + jarea - inter);
      if (iou > static_cast<float>(iou_threshold)) suppressed[static_cast<size_t>(j)] = 1;
    }
  }
  return keep.narrow(0, 0, num_keep).to(dets_in.device());
}

}  // namespace

// NLF ships as a TorchScript module (see nlf.hpp). We load and run it directly; the heavy
// torch::jit type lives here, kept out of the public header via this pimpl.
struct Nlf::Impl {
  torch::jit::script::Module module;
  at::Device device{at::kCPU};
  NlfConfig cfg;
};

Nlf Nlf::load(const std::string& torchscript_path, at::Device device, NlfConfig cfg) {
  NCG_CHECK(std::filesystem::exists(torchscript_path),
            "Nlf::load: TorchScript file not found: '{}'. Download the released NLF model "
            "(e.g. nlf_l_multi.torchscript) from github.com/isarandi/nlf.",
            torchscript_path);

  Nlf nlf;
  nlf.impl_ = std::make_shared<Impl>();
  nlf.impl_->device = device;
  nlf.impl_->cfg = std::move(cfg);

  // torchvision::nms is registered at static-init time (see TORCH_LIBRARY below), so NLF's
  // detector graph resolves it the moment the module loads.
  try {
    nlf.impl_->module = torch::jit::load(torchscript_path, device);
  } catch (const c10::Error& e) {
    NCG_THROW("Nlf::load: torch::jit::load failed for '{}': {}", torchscript_path, e.what());
  }
  nlf.impl_->module.eval();
  NCG_LOG_INFO("NLF: loaded TorchScript '{}' on {}", torchscript_path, device.str());
  return nlf;
}

NlfPrediction Nlf::detect(const Tensor& image_chw) const {
  NCG_CHECK(impl_ != nullptr, "Nlf::detect: model not loaded");
  NCG_CHECK(image_chw.dim() == 3 && image_chw.size(0) == 3,
            "Nlf::detect: expected a [3,H,W] image, got a {}-D tensor", image_chw.dim());
  const auto device = impl_->device;

  // NLF expects a uint8 RGB batch [B,3,H,W] on the model's device (per demo.ipynb:
  // `image = torchvision.io.read_image(...); frame_batch = image.unsqueeze(0)`).
  const auto frames =
      (image_chw.clamp(0.0, 1.0) * 255.0).round().to(at::kByte).unsqueeze(0).to(device);

  torch::NoGradGuard no_grad;
  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(frames);
  // model_name="smplx" => a 55-joint pose + 10 betas matching SmplxModel (default is "smpl").
  std::unordered_map<std::string, torch::jit::IValue> kwargs;
  kwargs["model_name"] = impl_->cfg.model_name;

  const auto result = impl_->module.get_method(impl_->cfg.method)(std::move(inputs), kwargs);
  NCG_CHECK(result.isGenericDict(),
            "Nlf::detect: '{}' did not return a dict; confirm the API with tools/dump_nlf.py",
            impl_->cfg.method);
  const auto dict = result.toGenericDict();

  // detect_smpl_batched returns per-key, per-image results for a multi-person detector. Each
  // value is a List (one entry per input image) of [num_detections, ...] tensors. We sent one
  // image and keep one detection. The exact nesting is confirmed by tools/dump_nlf.py; handle
  // the plausible shapes (TensorList / generic List / bare Tensor) defensively.
  const int det = impl_->cfg.detection;
  auto pick = [&](const char* key) -> Tensor {
    NCG_CHECK(dict.contains(key), "Nlf::detect: output missing key '{}'", key);
    const auto value = dict.at(key);
    Tensor per_image;
    if (value.isTensorList()) {
      per_image = value.toTensorList().get(0);
    } else if (value.isList()) {
      per_image = value.toList().get(0).toTensor();
    } else if (value.isTensor()) {
      per_image = value.toTensor();
    } else {
      NCG_THROW("Nlf::predict: unexpected IValue type for key '{}'", key);
    }
    NCG_CHECK(per_image.size(0) > det,
              "Nlf::predict: requested detection {} but only {} found for '{}'", det,
              per_image.size(0), key);
    return per_image.select(0, det).to(at::kCPU, at::kFloat).contiguous();
  };

  // pose may be flat [J*3] or [J,3]; reshape to [1,J,3] (SmplxModel::forward accepts both, and
  // J is inferred so this works whether NLF emits SMPL (24j) or SMPL-X (55j) — the consuming
  // SmplxModel must have a matching joint count; verify against the golden).
  NlfPrediction out;
  out.params.pose_aa = pick("pose").reshape({1, -1, 3}).contiguous();
  out.params.betas = pick("betas").reshape({1, -1}).contiguous();
  out.params.transl = pick("trans").reshape({1, 3}).contiguous();
  out.vertices2d = pick("vertices2d").contiguous();  // [V,2] image-space mesh projection
  out.vertices3d = pick("vertices3d").contiguous();  // [V,3] camera-space (z = depth)
  return out;
}

}  // namespace ncg::body

// Register our torchvision::nms so NLF's detector graph resolves it without torchvision's
// Python-entangled _C.so. The static initializer runs because nlf.o is linked into any target
// that uses Nlf (the catch-all kernel handles both CPU and CUDA inputs).
TORCH_LIBRARY(torchvision, m) {  // NOLINT
  m.def("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor", &ncg::body::nms_cpu);
}

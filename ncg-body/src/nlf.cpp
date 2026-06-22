#include <ncg/body/nlf.hpp>

#include <ncg/core/error.hpp>
#include <ncg/core/logging.hpp>

#include <torch/script.h>

#include <dlfcn.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ncg::body {

namespace {

// NLF's multi-person graph calls torchvision custom ops (torchvision::nms). Those are
// registered by static initializers inside torchvision's compiled library, so we dlopen it
// (RTLD_GLOBAL, so the symbols are visible to the torch dispatcher) before loading the module.
void ensure_torchvision_ops(const std::string& configured) {
  std::string path = configured;
  if (path.empty()) {
    if (const char* env = std::getenv("NCG_TORCHVISION_LIB")) path = env;
  }
  if (path.empty()) return;  // caller opted out (e.g. crop model with no torchvision ops)

  static void* handle = nullptr;  // load once per process
  if (handle != nullptr) return;
  handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
  NCG_CHECK(handle != nullptr,
            "Nlf: failed to load torchvision ops library '{}': {}. Set NlfConfig::ops_library "
            "or NCG_TORCHVISION_LIB to torchvision's _C/libtorchvision .so.",
            path, dlerror() ? dlerror() : "unknown");
  NCG_LOG_INFO("NLF: registered torchvision ops from {}", path);
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

  // NLF's multi-person graph references torchvision::nms — register those ops first.
  ensure_torchvision_ops(nlf.impl_->cfg.ops_library);

  try {
    nlf.impl_->module = torch::jit::load(torchscript_path, device);
  } catch (const c10::Error& e) {
    NCG_THROW("Nlf::load: torch::jit::load failed for '{}': {}", torchscript_path, e.what());
  }
  nlf.impl_->module.eval();
  NCG_LOG_INFO("NLF: loaded TorchScript '{}' on {}", torchscript_path, device.str());
  return nlf;
}

SmplxParams Nlf::predict(const Tensor& image_chw) const {
  NCG_CHECK(impl_ != nullptr, "Nlf::predict: model not loaded");
  NCG_CHECK(image_chw.dim() == 3 && image_chw.size(0) == 3,
            "Nlf::predict: expected a [3,H,W] image, got a {}-D tensor", image_chw.dim());
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
            "Nlf::predict: '{}' did not return a dict; confirm the API with tools/dump_nlf.py",
            impl_->cfg.method);
  const auto out = result.toGenericDict();

  // detect_smpl_batched returns per-key, per-image results for a multi-person detector. Each
  // value is a List (one entry per input image) of [num_detections, ...] tensors. We sent one
  // image and keep one detection. The exact nesting is confirmed by tools/dump_nlf.py; handle
  // the plausible shapes (TensorList / generic List / bare Tensor) defensively.
  const int det = impl_->cfg.detection;
  auto pick = [&](const char* key) -> Tensor {
    NCG_CHECK(out.contains(key), "Nlf::predict: output missing key '{}'", key);
    const auto value = out.at(key);
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
  SmplxParams p;
  p.pose_aa = pick("pose").reshape({1, -1, 3}).contiguous();
  p.betas = pick("betas").reshape({1, -1}).contiguous();
  p.transl = pick("trans").reshape({1, 3}).contiguous();
  return p;
}

}  // namespace ncg::body

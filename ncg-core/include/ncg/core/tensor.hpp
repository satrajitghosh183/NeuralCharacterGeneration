#pragma once

#include <ncg/core/error.hpp>

#include <torch/torch.h>

namespace ncg {

/// Project-wide alias. We deliberately do NOT wrap at::Tensor: every ported model is
/// torch::nn::Module-shaped, so fighting LibTorch's tensor is pure cost. Custom CUDA
/// kernels operate on raw pointers extracted from these.
using Tensor = at::Tensor;

inline at::TensorOptions f32_cpu() {
  return at::TensorOptions().dtype(at::kFloat).device(at::kCPU);
}

inline at::TensorOptions f32_cuda(int device = 0) {
  return at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, device);
}

/// Assert a tensor is contiguous, float32, and on CUDA — the contract custom kernels expect.
inline void require_cuda_f32_contiguous(const Tensor& t, const char* who) {
  NCG_CHECK(t.defined(), "{}: tensor is undefined", who);
  NCG_CHECK(t.is_cuda(), "{}: expected CUDA tensor, got {}", who, t.device().str());
  NCG_CHECK(t.scalar_type() == at::kFloat, "{}: expected float32, got {}", who,
            c10::toString(t.scalar_type()));
  NCG_CHECK(t.is_contiguous(), "{}: expected contiguous tensor", who);
}

}  // namespace ncg

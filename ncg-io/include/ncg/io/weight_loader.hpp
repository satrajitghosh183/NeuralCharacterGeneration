#pragma once

#include <ncg/core/tensor.hpp>
#include <ncg/io/safetensors.hpp>

#include <torch/torch.h>

#include <set>
#include <string>
#include <string_view>
#include <unordered_map>

namespace ncg::io {

/// Name-based access to a safetensors file for porting pretrained weights into C++ modules.
/// The single place where PyTorch<->C++ parameter-name drift is reconciled.
class WeightMap {
public:
  explicit WeightMap(SafeTensors st);

  bool has(std::string_view name) const;

  /// Fetch a tensor, asserting its shape; returns a contiguous CPU copy in `dtype`.
  Tensor get(const std::string& name, at::IntArrayRef expected_shape,
             at::ScalarType dtype = at::kFloat);

  /// Copies file tensors into every parameter and buffer of `module`.
  /// `remap` maps a module param/buffer name -> file tensor name (identity if absent).
  /// When `strict`, asserts every module param/buffer was filled AND every file tensor was
  /// consumed -- the guard against silently-uninitialized or misnamed weights.
  void load_into(torch::nn::Module& module,
                 const std::unordered_map<std::string, std::string>& remap = {},
                 bool strict = true);

  /// Names present in the file but not yet consumed by a load_into call.
  std::vector<std::string> unconsumed() const;

private:
  SafeTensors st_;
  std::set<std::string> consumed_;
};

}  // namespace ncg::io

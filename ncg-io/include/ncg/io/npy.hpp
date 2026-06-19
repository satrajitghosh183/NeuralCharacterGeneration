#pragma once

#include <ncg/core/tensor.hpp>

#include <string>

namespace ncg::io {

/// Loads a NumPy .npy file as a CPU tensor (dtype inferred from the header). Used to read
/// golden reference tensors dumped by tools/dump_golden.py. C-order only.
Tensor load_npy(const std::string& path);

/// Saves a CPU tensor as a C-order .npy file.
void save_npy(const std::string& path, const Tensor& t);

}  // namespace ncg::io

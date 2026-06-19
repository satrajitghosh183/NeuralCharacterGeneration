#pragma once

#include <ncg/core/tensor.hpp>

#include <string>

namespace ncg::io {

/// Loads an image as a float32 CHW tensor in [0, 1] on CPU.
/// `desired_channels` forces 3 (RGB) or 4 (RGBA); 0 keeps the file's channel count.
Tensor load_image(const std::string& path, int desired_channels = 3);

/// Saves a float32 CHW tensor in [0, 1] (3 or 4 channels) as an 8-bit PNG.
void save_png(const std::string& path, const Tensor& image_chw);

}  // namespace ncg::io

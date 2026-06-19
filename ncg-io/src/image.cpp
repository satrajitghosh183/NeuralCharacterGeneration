#include <ncg/io/image.hpp>

#include <ncg/core/error.hpp>

#include <stb_image.h>
#include <stb_image_write.h>

#include <cstdint>
#include <vector>

namespace ncg::io {

Tensor load_image(const std::string& path, int desired_channels) {
  int w = 0;
  int h = 0;
  int file_channels = 0;
  // stb returns HWC uint8.
  unsigned char* data = stbi_load(path.c_str(), &w, &h, &file_channels, desired_channels);
  NCG_CHECK(data != nullptr, "load_image: failed to read '{}': {}", path, stbi_failure_reason());

  const int c = (desired_channels > 0) ? desired_channels : file_channels;
  // Wrap (copy) into a uint8 HWC tensor, then convert to float CHW in [0,1].
  Tensor hwc = at::from_blob(data, {h, w, c}, at::TensorOptions().dtype(at::kByte)).clone();
  stbi_image_free(data);

  Tensor chw = hwc.permute({2, 0, 1}).contiguous().to(at::kFloat).div_(255.0);
  return chw;
}

void save_png(const std::string& path, const Tensor& image_chw) {
  NCG_CHECK(image_chw.dim() == 3, "save_png: expected CHW tensor, got dim {}", image_chw.dim());
  const int64_t c = image_chw.size(0);
  NCG_CHECK(c == 1 || c == 3 || c == 4, "save_png: channels must be 1/3/4, got {}", c);

  Tensor hwc = image_chw.detach()
                   .to(at::kCPU, at::kFloat)
                   .clamp(0.0, 1.0)
                   .mul(255.0)
                   .round()
                   .to(at::kByte)
                   .permute({1, 2, 0})
                   .contiguous();

  const int h = static_cast<int>(hwc.size(0));
  const int w = static_cast<int>(hwc.size(1));
  const int stride = w * static_cast<int>(c);
  const int ok = stbi_write_png(path.c_str(), w, h, static_cast<int>(c), hwc.data_ptr<uint8_t>(),
                                stride);
  NCG_CHECK(ok != 0, "save_png: failed to write '{}'", path);
}

}  // namespace ncg::io

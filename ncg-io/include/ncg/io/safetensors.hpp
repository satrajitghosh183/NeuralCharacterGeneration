#pragma once

#include <ncg/core/tensor.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ncg::io {

/// Reader for the safetensors format (https://github.com/huggingface/safetensors):
///   [u64 little-endian header length][JSON header][raw tensor bytes].
/// Backed by mmap; views are zero-copy and valid only while this object is alive.
class SafeTensors {
public:
  /// Opens and mmaps the file. Throws on I/O or format error.
  static SafeTensors open(const std::string& path);

  SafeTensors(SafeTensors&&) noexcept;
  SafeTensors& operator=(SafeTensors&&) noexcept;
  SafeTensors(const SafeTensors&) = delete;
  SafeTensors& operator=(const SafeTensors&) = delete;
  ~SafeTensors();

  std::vector<std::string> names() const;
  bool has(const std::string& name) const;

  /// Zero-copy view onto the mmap'd bytes (valid while this SafeTensors lives).
  Tensor view(const std::string& name) const;

  /// Contents of the optional "__metadata__" entry.
  const std::map<std::string, std::string>& metadata() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  explicit SafeTensors(std::unique_ptr<Impl> impl);
};

/// Writes a safetensors file from name -> CPU contiguous tensor. Used by tests and the
/// offline export tooling. `metadata` is stored under "__metadata__".
void write_safetensors(const std::string& path, const std::map<std::string, Tensor>& tensors,
                       const std::map<std::string, std::string>& metadata = {});

}  // namespace ncg::io

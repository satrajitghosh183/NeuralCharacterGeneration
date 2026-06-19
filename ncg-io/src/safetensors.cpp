#include <ncg/io/safetensors.hpp>

#include <ncg/core/error.hpp>

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <fstream>

namespace ncg::io {
namespace {

using json = nlohmann::json;

struct DtypeInfo {
  at::ScalarType type;
  int64_t elem_size;
};

DtypeInfo dtype_from_string(const std::string& s) {
  if (s == "F64") return {at::kDouble, 8};
  if (s == "F32") return {at::kFloat, 4};
  if (s == "F16") return {at::kHalf, 2};
  if (s == "BF16") return {at::kBFloat16, 2};
  if (s == "I64") return {at::kLong, 8};
  if (s == "I32") return {at::kInt, 4};
  if (s == "I16") return {at::kShort, 2};
  if (s == "I8") return {at::kChar, 1};
  if (s == "U8") return {at::kByte, 1};
  if (s == "BOOL") return {at::kBool, 1};
  NCG_THROW("safetensors: unsupported dtype '{}'", s);
}

const char* dtype_to_string(at::ScalarType t) {
  switch (t) {
    case at::kDouble: return "F64";
    case at::kFloat: return "F32";
    case at::kHalf: return "F16";
    case at::kBFloat16: return "BF16";
    case at::kLong: return "I64";
    case at::kInt: return "I32";
    case at::kShort: return "I16";
    case at::kChar: return "I8";
    case at::kByte: return "U8";
    case at::kBool: return "BOOL";
    default: NCG_THROW("safetensors: cannot serialize dtype {}", c10::toString(t));
  }
}

}  // namespace

struct SafeTensors::Impl {
  int fd = -1;
  void* addr = nullptr;
  size_t map_size = 0;
  const uint8_t* data_base = nullptr;  // start of tensor byte region

  struct Entry {
    at::ScalarType dtype;
    std::vector<int64_t> shape;
    size_t begin = 0;
    size_t end = 0;
  };
  std::map<std::string, Entry> entries;
  std::map<std::string, std::string> metadata;

  ~Impl() {
    if (addr != nullptr && addr != MAP_FAILED) ::munmap(addr, map_size);
    if (fd >= 0) ::close(fd);
  }
};

SafeTensors::SafeTensors(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SafeTensors::SafeTensors(SafeTensors&&) noexcept = default;
SafeTensors& SafeTensors::operator=(SafeTensors&&) noexcept = default;
SafeTensors::~SafeTensors() = default;

SafeTensors SafeTensors::open(const std::string& path) {
  auto impl = std::make_unique<Impl>();

  impl->fd = ::open(path.c_str(), O_RDONLY);
  NCG_CHECK(impl->fd >= 0, "safetensors: cannot open '{}': {}", path, std::strerror(errno));

  struct stat st {};
  NCG_CHECK(::fstat(impl->fd, &st) == 0, "safetensors: fstat failed for '{}'", path);
  impl->map_size = static_cast<size_t>(st.st_size);
  NCG_CHECK(impl->map_size >= 8, "safetensors: file '{}' too small", path);

  impl->addr = ::mmap(nullptr, impl->map_size, PROT_READ, MAP_PRIVATE, impl->fd, 0);
  NCG_CHECK(impl->addr != MAP_FAILED, "safetensors: mmap failed for '{}': {}", path,
            std::strerror(errno));

  const auto* base = static_cast<const uint8_t*>(impl->addr);
  uint64_t header_len = 0;
  std::memcpy(&header_len, base, sizeof(header_len));  // little-endian on x86/arm64
  NCG_CHECK(8 + header_len <= impl->map_size, "safetensors: header length {} exceeds file size",
            header_len);

  const char* header_begin = reinterpret_cast<const char*>(base + 8);
  impl->data_base = base + 8 + header_len;

  json header = json::parse(header_begin, header_begin + header_len);
  const size_t data_region = impl->map_size - (8 + header_len);

  for (auto it = header.begin(); it != header.end(); ++it) {
    if (it.key() == "__metadata__") {
      for (auto m = it->begin(); m != it->end(); ++m) {
        impl->metadata[m.key()] = m->get<std::string>();
      }
      continue;
    }
    const json& e = it.value();
    Impl::Entry entry;
    entry.dtype = dtype_from_string(e.at("dtype").get<std::string>());
    entry.shape = e.at("shape").get<std::vector<int64_t>>();
    const auto offsets = e.at("data_offsets").get<std::vector<size_t>>();
    NCG_CHECK(offsets.size() == 2, "safetensors: bad data_offsets for '{}'", it.key());
    entry.begin = offsets[0];
    entry.end = offsets[1];
    NCG_CHECK(entry.end <= data_region && entry.begin <= entry.end,
              "safetensors: offsets out of range for '{}'", it.key());
    impl->entries.emplace(it.key(), std::move(entry));
  }

  return SafeTensors(std::move(impl));
}

std::vector<std::string> SafeTensors::names() const {
  std::vector<std::string> out;
  out.reserve(impl_->entries.size());
  for (const auto& [k, v] : impl_->entries) out.push_back(k);
  return out;
}

bool SafeTensors::has(const std::string& name) const {
  return impl_->entries.find(name) != impl_->entries.end();
}

Tensor SafeTensors::view(const std::string& name) const {
  auto it = impl_->entries.find(name);
  NCG_CHECK(it != impl_->entries.end(), "safetensors: no tensor named '{}'", name);
  const auto& e = it->second;
  auto opts = at::TensorOptions().dtype(e.dtype).device(at::kCPU);
  // Cast away const: the view is read-only by contract (callers clone before mutating).
  void* ptr = const_cast<uint8_t*>(impl_->data_base + e.begin);
  return at::from_blob(ptr, at::IntArrayRef(e.shape), opts);
}

const std::map<std::string, std::string>& SafeTensors::metadata() const { return impl_->metadata; }

void write_safetensors(const std::string& path, const std::map<std::string, Tensor>& tensors,
                       const std::map<std::string, std::string>& metadata) {
  json header = json::object();
  if (!metadata.empty()) {
    json meta = json::object();
    for (const auto& [k, v] : metadata) meta[k] = v;
    header["__metadata__"] = meta;
  }

  // Assign sequential byte offsets in (sorted) map order; collect contiguous CPU tensors.
  std::vector<Tensor> ordered;
  size_t offset = 0;
  for (const auto& [name, t] : tensors) {
    NCG_CHECK(t.device().is_cpu(), "write_safetensors: '{}' must be CPU", name);
    Tensor c = t.contiguous();
    const int64_t bytes = c.numel() * c.element_size();
    json entry;
    entry["dtype"] = dtype_to_string(c.scalar_type());
    entry["shape"] = std::vector<int64_t>(c.sizes().begin(), c.sizes().end());
    entry["data_offsets"] = std::vector<size_t>{offset, offset + static_cast<size_t>(bytes)};
    header[name] = entry;
    offset += static_cast<size_t>(bytes);
    ordered.push_back(std::move(c));
  }

  const std::string header_str = header.dump();
  uint64_t header_len = header_str.size();

  std::ofstream os(path, std::ios::binary);
  NCG_CHECK(os.good(), "write_safetensors: cannot open '{}' for writing", path);
  os.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
  os.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));
  for (const auto& c : ordered) {
    os.write(static_cast<const char*>(c.const_data_ptr()),
             static_cast<std::streamsize>(c.numel() * c.element_size()));
  }
  NCG_CHECK(os.good(), "write_safetensors: write error for '{}'", path);
}

}  // namespace ncg::io

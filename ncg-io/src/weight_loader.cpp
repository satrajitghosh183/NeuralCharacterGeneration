#include <ncg/io/weight_loader.hpp>

#include <ncg/core/error.hpp>
#include <ncg/core/logging.hpp>

namespace ncg::io {
namespace {

std::string shape_str(at::IntArrayRef s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    out += std::to_string(s[i]);
    if (i + 1 < s.size()) out += ", ";
  }
  return out + "]";
}

}  // namespace

WeightMap::WeightMap(SafeTensors st) : st_(std::move(st)) {}

bool WeightMap::has(std::string_view name) const { return st_.has(std::string(name)); }

Tensor WeightMap::get(const std::string& name, at::IntArrayRef expected_shape,
                      at::ScalarType dtype) {
  NCG_CHECK(st_.has(name), "WeightMap: missing tensor '{}'", name);
  Tensor view = st_.view(name);
  NCG_CHECK(view.sizes() == expected_shape, "WeightMap: '{}' shape {} != expected {}", name,
            shape_str(view.sizes()), shape_str(expected_shape));
  consumed_.insert(name);
  return view.to(dtype).contiguous();
}

void WeightMap::load_into(torch::nn::Module& module,
                          const std::unordered_map<std::string, std::string>& remap, bool strict) {
  torch::NoGradGuard no_grad;

  auto fill = [&](const std::string& mod_name, Tensor dst) {
    auto it = remap.find(mod_name);
    const std::string file_name = (it != remap.end()) ? it->second : mod_name;
    NCG_CHECK(st_.has(file_name), "WeightMap: module needs '{}' (file key '{}') which is absent",
              mod_name, file_name);
    Tensor src = st_.view(file_name);
    NCG_CHECK(src.sizes() == dst.sizes(),
              "WeightMap: '{}' (file '{}') shape {} != module param {}", mod_name, file_name,
              shape_str(src.sizes()), shape_str(dst.sizes()));
    dst.copy_(src.to(dst.device(), dst.scalar_type()));
    consumed_.insert(file_name);
  };

  for (const auto& p : module.named_parameters(/*recurse=*/true)) fill(p.key(), p.value());
  for (const auto& b : module.named_buffers(/*recurse=*/true)) fill(b.key(), b.value());

  if (strict) {
    const auto leftover = unconsumed();
    NCG_CHECK(leftover.empty(), "WeightMap: {} file tensor(s) unconsumed (first: '{}'). "
                                "Name mismatch or wrong model.",
              leftover.size(), leftover.front());
  }
  NCG_LOG_DEBUG("WeightMap: loaded {} tensors into module", consumed_.size());
}

std::vector<std::string> WeightMap::unconsumed() const {
  std::vector<std::string> out;
  for (const auto& n : st_.names()) {
    if (consumed_.find(n) == consumed_.end()) out.push_back(n);
  }
  return out;
}

}  // namespace ncg::io

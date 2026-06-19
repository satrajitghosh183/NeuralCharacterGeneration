#include <ncg/body/nlf.hpp>

#include <ncg/core/error.hpp>

namespace ncg::body {

// PORT STATUS: see nlf.hpp. These are intentional, loud stubs so the rest of the pipeline
// compiles and runs (via neutral / file-loaded params) while NLF is ported and parity-tested.

Nlf Nlf::load(const std::string& weights_path, at::Device /*device*/) {
  NCG_THROW("Nlf::load: NLF port not implemented yet (weights '{}'). Vendor isarandi/nlf, "
            "define the module here, then load via io::WeightMap and validate with "
            "tests/golden/test_golden_nlf.cpp. See docs/parity.md.",
            weights_path);
}

SmplxParams Nlf::predict(const Tensor& /*image_chw*/) const {
  NCG_NOT_IMPLEMENTED();
}

}  // namespace ncg::body

# Parity harness self-test + (when golden data is present) real per-layer model parity.
# nlohmann_json is needed by golden_fixture.hpp to parse manifests.
ncg_add_test(test_golden_linear
  SOURCES test_golden_linear.cpp
  DEPS    ncg::io nlohmann_json::nlohmann_json
  LABELS  "golden")

# Real model parity tests SKIP at runtime when their data/golden/<model> dir is absent.
ncg_add_test(test_golden_nlf
  SOURCES test_golden_nlf.cpp
  DEPS    ncg::io ncg::body nlohmann_json::nlohmann_json
  LABELS  "golden")
